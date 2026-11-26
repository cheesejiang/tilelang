/*!
 * \file ascend/costmodel/simtvf/model/ub_access.cc
 * \brief Pack SIMT UB requests, then price their ideal service path.
 *
 * Address compatibility and entry capacity determine packing; full-line
 * identity is enforced during insertion. Latency adds serial entry publishing
 * to the intrinsic base under the assumption of overlapping response stages.
 * Independent instructions share that pipeline instead of summing latency.
 * See ub_access.h for the input contract, ideal-service assumptions and
 * unknown-cost cases.
 */
#include "ub_access.h"
#include <algorithm>
#include <limits>

namespace tvm::tl::ascend::costmodel::simtvf {

namespace {
constexpr auto &geometry = kDav3510UbGeometry;

// DAV-3510 SIMT UB request packing. Each cell below represents an 8B block;
// the numbers are bank keys, and columns are virtual groups:
//
//   Byte range       group 0   group 1   ...   group 15
//     0 .. 127          0         1      ...      15
//   128 .. 255         16        17      ...      31
//   256 .. 383         32        33      ...      47
//   384 .. 511         48        49      ...      63
//   512 .. 639          0         1      ...      15   (next 512B span)
//       ...
//
//   key = (address >> 3) & 63; group = key & 15; line = address & ~7
//
//   warp request (32 lanes, <=4B/lane)
//       -> 2B elements -> entry 0 [group 0 | group 1 | ... | group 15]
//                        entry 1 [group 0 | group 1 | ... | group 15]
//                        ...
// Each entry accepts one key per group and at most 64 charged elements,
// including repeated elements. A matching key also requires the same full
// 8B line: addresses 0 and 4 can merge, but 0 and 512 cannot. Different keys
// in the same group (e.g. 0 and 16) occupy different entries.
// Requests are packed independently; these entry counts describe request
// structure rather than execution cycles.
struct Entry {
  int elements{0};
  uint32_t groups{0};
  std::array<int64_t, kDav3510UbGeometry.bank_keys> lines;
  std::array<uint8_t, kDav3510UbGeometry.virtual_groups> masks{};
  Entry() { lines.fill(-1); }
};

} // namespace

// Request packing: insert one warp's <=4B-per-lane slice into entries under
// the key/group/line and capacity rules above. Inactive lanes contribute
// nothing. Record incomplete 4B store coverage per entry for the pricing
// stage; this stage describes request geometry without assigning cycles.
std::optional<SimtUbRequestFeatures>
AnalyzeSimtUbRequest(const SimtUbLaneAddresses &addresses, int bytes,
                     bool store) {
  if (bytes < 1 || bytes > geometry.request_bytes)
    return std::nullopt;
  for (auto address : addresses)
    if (address && (*address < 0 ||
                    *address > std::numeric_limits<int64_t>::max() - bytes))
      return std::nullopt;
  SimtUbRequestFeatures cost;
  std::vector<Entry> entries;
  // Process descending active lanes, ascending 2B elements
  // inside one lane's <=4B slice. Never concatenate independent slices.
  for (int lane = geometry.warp_size - 1; lane >= 0; --lane) {
    // lane 31 → lane 30 → … → lane 0;   然后每个 lane 内，按地址从低到高处理 fragment
    if (!addresses[lane])
      continue;
    int64_t begin = *addresses[lane];
    for (int offset = 0; offset < bytes;) {
      int64_t address = begin + offset;
      int count = // geometry.element_bytes=2，fragment按照2B边界对齐
          std::min(bytes - offset, geometry.element_bytes -
                                       int(address % geometry.element_bytes));
      int key = (address / geometry.line_bytes) % geometry.bank_keys;
      int group = key % geometry.virtual_groups;
      int64_t line = address & ~int64_t(geometry.line_bytes - 1);
      auto it = std::find_if(entries.begin(), entries.end(), [&](Entry &e) {
        // elements;   // 已计入多少个 fragment
        // groups;     // 哪些 virtual group 已占用
        // lines[key]; // 某个 key 对应的完整 line，初始为 -1
        // masks[group]; // 已覆盖哪些字，供 store 检查使用 例如：mask=0b00001111，表示该 entry 的 group0-4 已覆盖
        if (e.elements >= geometry.entry_elements) // 64 个 fragment 了，不能再加了
          return false;
        if (e.lines[key] != -1)
          return e.lines[key] == line; // 该 key 已经存在时，必须匹配相同完整 line
        return !(e.groups & (1U << group)); // also bounds keys to 16 key 尚不存在时，要求对应 group 未被其他 key 占据。
      });
      if (it == entries.end()) { // 如果没有找到合适的 entry，就新建一个 entry
        entries.emplace_back();
        it = entries.end() - 1;
      }
      ++it->elements; // 更新 entry 的 fragment 数量
      it->lines[key] = line;
      it->groups |= 1U << group;
      it->masks[group] |= ((1U << count) - 1)
                          << (address % geometry.line_bytes);
      offset += count;
    }
  }
  if (entries.empty())
    return cost;
  cost.requests = 1;
  cost.entries += entries.size();
  if (store) {
    for (const auto &entry : entries) {
      bool partial = false;
      for (uint8_t mask : entry.masks)
        for (int shift = 0; shift < geometry.line_bytes;
             shift += geometry.store_word_bytes) {
          int word_mask = (1 << geometry.store_word_bytes) - 1;
          int coverage = (mask >> shift) & word_mask;
          partial |= coverage != 0 && coverage != word_mask;
        }
      cost.partial += partial; // 统计的是存在 Partial Word 的 Entry 数量，而不是 Partial Word 的总数量
    }
  }
  return cost;
}

// Instruction decomposition: accept a supplied machine width and pack its
// <=4B slices independently. An active b64 instruction has two requests but
// remains one instruction; its intrinsic base is charged once during pricing.
std::optional<SimtUbInstructionFeatures>
AnalyzeSimtUbInstruction(const SimtUbLaneAddresses &addresses, int bytes,
                         bool store) {
  if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
    return std::nullopt;
  SimtUbInstructionFeatures result;
  result.instructions = 1;
  for (auto address : addresses) {
    if (address && (*address < 0 ||
                    *address > std::numeric_limits<int64_t>::max() - bytes))
      return std::nullopt;
  }
  for (int offset = 0; offset < bytes; offset += geometry.request_bytes) {// 若byte=8，那么offset=0,4，分别计算两次请求
    auto slice = addresses; // 复制一份地址数组
    for (auto &address : slice)
      if (address)
        *address += offset; // 每个 lane 的地址加上偏移量，得到每个 lane 的请求地址(也是拆分成多个请求的地址)
    auto part = AnalyzeSimtUbRequest( // 当前片段的每个Lane的访存地址，当前片段实际的字节宽度
        slice, std::min(geometry.request_bytes, bytes - offset), store);
    if (!part)
      return std::nullopt;
    result.entries += part->entries; // 累加分析结果
    result.partial += part->partial;
    result.requests += part->requests;
  }
  return result;
}

// Instruction pricing: ideal latency is the LDS/STS intrinsic base plus one
// entry-issue interval for every entry after the first, across all slices.
// Response stages overlap within this service model. An inactive instruction
// costs zero; a partial store keeps its features but has unknown latency
// because the required read-modify-write dependency is not modeled.
std::optional<SimtUbInstructionCost>
EstimateSimtUbInstructionLatency(const SimtUbLaneAddresses &addresses,
                                 int bytes, bool store) {
  auto features = AnalyzeSimtUbInstruction(addresses, bytes, store);
  if (!features)
    return std::nullopt;
  SimtUbInstructionCost result;
  static_cast<SimtUbInstructionFeatures &>(result) = *features;
  if (features->requests == 0) {
    result.estimated_latency = 0;
    result.service_cycles = 0;
    return result;
  }
  // Preserve geometry while leaving the unmodeled RMW latency unknown.
  if (features->partial != 0)
    return result;
  // 对于活跃 load 或无 partial 的 store
  const auto &instruction = store ? kDav3510StsCost : kDav3510LdsCost;
  // The base includes the first entry; charge only additional entry issues.
  // 基础指令成本已经包含第一个 Entry 的发布成本。所以只需要额外计算剩余 Entry 的发布成本
  double publish_cycles = (features->entries - 1) * kDav3510UbEntryIssueCycles; 
  result.service_cycles = // 所有entry的发布占用 entry总数*发布间隔
      double(features->entries) * kDav3510UbEntryIssueCycles;
  result.estimated_latency =
      AddCycles(EstimateInstructionCycles(instruction, 1), publish_cycles);
      // 估计指令周期 + 额外的 entry 发布占用
  return result;
}

// Batch completion: consecutive independent instructions occupy consecutive
// publishing slots, without waiting for earlier responses. For n identical
// instances, the last starts after (n-1)*service_cycles, then takes one latency
// to complete. Track the publisher separately from the latest completion.
// No concurrency divisor or fitted queue penalty is needed. Missing cost or
// arithmetic overflow makes the batch unknown.
// SimtUbInstructionAccess { 数据包含
//     addresses,   // 32 个 lane 各自的起始字节地址，空值代表 inactive
//     bytes,        // 每个 lane 读取/写入多少字节
//     repetitions  // 这类独立指令实例重复多少次
// }
std::optional<SimtUbAccessCost> EstimateSimtUbAccessLatency( // 接收一组同方向的独立指令，计算它们的总延迟
    const std::vector<SimtUbInstructionAccess> &instructions, bool store) {
  SimtUbAccessCost total;
  total.estimated_latency = 0;
  total.service_cycles = 0;
  for (const auto &instruction : instructions) { // 遍历每条指令，得到每条指令的单次执行成本
    auto cost = EstimateSimtUbInstructionLatency(instruction.addresses,
                                                 instruction.bytes, store);
    if (!cost || !cost->estimated_latency || !cost->service_cycles ||
        instruction.repetitions <= 0)
      return std::nullopt;
    auto accumulate = [&](int64_t &sum, int64_t count) { // 局部函数，在后面被调用
      if (count >
          (std::numeric_limits<int64_t>::max() - sum) / instruction.repetitions) // 判断是否溢出
        return false;
      sum += count * instruction.repetitions;
      return true;
    };
    if (!accumulate(total.instructions, cost->instructions) ||
        !accumulate(total.requests, cost->requests) ||
        !accumulate(total.entries, cost->entries) ||
        !accumulate(total.partial, cost->partial))
      return std::nullopt;
    // Inactive instructions contribute no publishing work or completion tail.
    if (cost->requests == 0)
      continue;
    auto last_start =
        AddCycles(total.service_cycles, // P + (n - 1) × S
                  EstimateInstructionCycles({cost->service_cycles},
                                            instruction.repetitions - 1));
    auto completion = AddCycles(last_start, cost->estimated_latency);
    // 这一类指令的完成时间 = P + (n - 1) × S + L

    // 更新发布所需时间 P + n × S                
    total.service_cycles =
        AddCycles(total.service_cycles,
                  EstimateInstructionCycles({cost->service_cycles},
                                            instruction.repetitions));
    if (!completion || !total.service_cycles)
      return std::nullopt; 
    total.estimated_latency = std::max(*total.estimated_latency, *completion);
    // 更新整个 batch 的最晚完成时间
  } 
  return total;
}

} // namespace tvm::tl::ascend::costmodel::simtvf
