# TileKernels 的 padding 候选核查

检查的是本地 `TileKernels/tile_kernels`，commit `36d9e45`。检索转置、量化、
MoE、Engram、mHC 的 shared 访问，阅读下列重点实现及对应测试。
该库 README 面向 NVIDIA SM90/SM100；以下不是现成 Ascend kernel，也没有新的
Ascend 实测加速证据。源码已有手工优化必须纳入对照，不能删掉后只报告改善。

## 优先方向

| 优先级 | 实现 | 实际 shared 使用 | 比之前列写回更值得试的原因 | 主要限制 |
| --- | --- | --- | --- | --- |
| 1 | [batched_transpose_kernel.py](../../../../TileKernels/tile_kernels/transpose/batched_transpose_kernel.py) | 每线程读 4×4 微块，在寄存器内转置，写 padded shared，再由另一种固定线程布局读出 | shared 是真正的线程间重排通道；每个输出元素经过它，没有 QK/GEMM 或全局归约包围一小段列写回 | 原实现已有 `+4 elements` 和行旋转；Ascend 固定 +32B 不一定比其更好；手工线程绑定、local 临时值、barrier、直接 GM 访问不在当前 pass 完整支持范围 |
| 2 | [per_channel_cast_and_transpose_kernel.py](../../../../TileKernels/tile_kernels/quant/per_channel_cast_and_transpose_kernel.py) | BF16 输入先转置进 `out_shared[64,132]`，分 4 阶段读入 fragment，求 absmax 并输出转置 FP8 和 scale | 转置暂存是每个元素都走的真实生产链，且有按通道归约约束 | 原有 +4 BF16（8B）padding 和行旋转已优化；分片 copy、转换、scale 函数、固定线程和动态尺寸需移植/扩展；不能省略量化后宣称完整算子受益 |
| 3 | [swiglu_forward_and_per_channel_cast_and_transpose_kernel.py](../../../../TileKernels/tile_kernels/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py) 的 transpose 分支 | SwiGLU 在寄存器产生值，按转置方向写 `act_shared`，再读出求 scale、转 FP8 | 中间值直接来自计算，避免了预计算 logits 的人工交接；源码明确讨论冲突与 swizzle 成本 | 源码第 151 行明确选择接受 4x bank conflict，因为 swizzle 更贵；这是 CUDA 取舍，不是 Ascend 收益证明。exp、转换可能掩盖 shared 改善，优先级低于独立 transpose |

注意：transpose 本身也是“完整 tile”。它与前面的独立逐元素 scale 不同：输入、
输出在 GM 中的连续方向不同，shared 用来在两种线程组织间交换数据。只把线程
统一排成某一方向连续，并不能自动保证两端 GM 都合并访问；ownership 仍可能
改善，但必须同时检查生产者、消费者和 GM 流量，不能先断言它无效。

## 独立 transpose 的源码证据

`batched_transpose_kernel.py`：

- 第 45 行：`out_shared = alloc_shared((block_y, block_x + 4), dtype)`。
- 第 59～63 行：连续读取 GM 的 4×4 微块，在每线程寄存器中转置。
- 第 67～69 行：`swizzle_j=(j+tid//(8//dtype.bytes))%4`，按转置方向写 shared。
- 第 71 行：barrier，表明接下来可能由其他线程读取。
- 第 73～74 行：显式 vec4 `loop_layout` 读 shared 并连续写 GM。

已有 [transpose 测试](../../../../TileKernels/tests/transpose/test_transpose.py)
覆盖 BF16/FP8，batched 版本还覆盖 FP32，并包含 benchmark 和非连续输入 stride。
这些是可复用的正确性用例，不是已提供的 Ascend 性能数字。

## 用现有 UB 模型做的地址预筛选

保持源码的生产者行旋转、256 线程、4×4 微块和 vec4 消费者布局不变；分析
满 tile 中全部 warp 的 shared store + shared load，比较三类物理地址：

1. 无 padding：`(r*block_x+c)*B`。
2. 源码手工方案：`(r*(block_x+4)+c)*B`。
3. 固定 32B 周期方案：`(r*block_x+c)*B + floor(r/K)*32`。

对第三类枚举 `1 <= K < block_y`，先最小化总 entry，成本相同再最小化内部空洞
字节数。没有引入额外重复计算。BF16 vec4 作为 b64；FP32 vec4 按两个 b64
机器访问估计，再由模型拆 request。实际 backend 指令需移植后核对。

| tile `(block_x,block_y)` | dtype | 无 padding entry | 源码 +4 elements entry | 最佳 K（行） | 固定 32B entry | 固定 32B 额外空间 |
| --- | --- | --- | --- | --- | --- | --- |
| 128×128 | BF16 | 8704 | 1024 | 32 | 2560 | 96B |
| 128×128 | FP32 | 18432 | 4096 | 32 | 6144 | 96B |
| 128×64 | BF16 | 2304 | 512 | 16 | 768 | 96B |
| 128×64 | FP32 | 5120 | 2048 | 16 | 2048 | 96B |

用当前 `ub_access.cc` 的独立 C++ 模型检查了 64/128 两轴组合及 1/2/4B 元素的
12 组地址几何。上表不是 CANN 编译、不是设备测试，也不包含 GM/DMA、同步、
地址计算、寄存器和转换成本。候选族外的其他布局没有被穷举。

由此只能得出：

- 对这些保留原线程安排的地址几何，32B 周期 padding 相比无 padding 有模型改善。
- 原始 CUDA 小 padding + rotation 经常比固定 32B 族更省 entry，不能忽略。
- 固定 32B 方案的空间开销小，例如 128×128 BF16 为 96B；原源码的 +4 BF16
  allocation 则额外占 128×4×2=1024B，但 row pitch=264B，不能直接视为当前
  Ascend DMA lowering 的合法方案。
- 有真实优化对象，不代表显著整核收益；DMA、同步仍可能使结果接近。

## 当前 pass 的支持缺口与移植方式

| 环节 | 当前限制 | 合理处理 |
| --- | --- | --- |
| CUDA launch、GM 访问 | 多维 grid、thread binding 和 CUDA barrier 不能原样套 Ascend；Parallel 直接 GM 访问被当前资格检查排除 | 将 grid 展平，使用 Ascend SIMT 与适当搬运/同步；端到端计入新增 staging 成本 |
| 手工微块计算 | 源码用 local 临时数组、串行/vectorized 循环和标量绑定，未必被收集为可改写 ParallelOp | 优先保持真实计算与线程分工做显式物理布局对照；不能删除使用者让 pass 误认为安全 |
| 行旋转与 stride | `col*4 + swizzle_j` 不一定给出等差活跃地址；当前 hint 只支持规则等差 | 可按有限 j/warp 相位提取结构化候选，再用完整实际地址评分；不能只拿一对 lane 外推 |
| 两套线程布局 | 来源的 producer/consumer 不同，GM 两端都需要考虑 | baseline、ownership、padding、auto 用相同输入 IR；若先做手工对照，应明确它不等同自动 pass 选择 |
| 现有显式 padding | 原 +4 elements 是用户物理布局；当前 pass 应尊重而非覆盖 | 建立逻辑 tile 与独立物理布局表示；原源码方案另作参考，不能当成未优化 baseline |
| 量化和融合 | fragment slice、absmax、exp、FP8/scale 转换可能超出当前模型或资格 | 独立 transpose 验证后，再逐步保留完整计算链；确认每次扩展的地址与精度 |

优先做独立 transpose 的完整、非重复工作量对照。先查看实际 ISA/活跃 lane，
比较 GM→结果的整核时延；若仍无稳定加速，再决定是否值得继续融合版。

## 排除或降低优先级的其他 kernel

| 实现 | 核查结果 |
| --- | --- |
| [per_channel_cast_fused_kernel.py](../../../../TileKernels/tile_kernels/quant/per_channel_cast_fused_kernel.py) | `amax_shared[col_id%VEC_K, i]` 看似跨行，但同一 warp 中 row 固定、i 从 lane 开始，lane 间实际连续；后续 scale 读取也连续。不能凭形状认定有冲突 |
| [moe/common.py / get_topk_group_idx](../../../../TileKernels/tile_kernels/moe/common.py) | 固定 lane 负责一个 expert group，确有分组读取；但已用 `(i+lane)%num_vec_experts_per_group` rotation 避免冲突。应先保留原 rotation 检查实际成本，不能删掉它制造 padding 收益 |
| [get_fused_mapping_kernel.py](../../../../TileKernels/tile_kernels/moe/get_fused_mapping_kernel.py) | shared histogram 有数据相关 atomic；同 expert 的地址竞争不会被 padding 消除；跨 SM/grid 同步又增加移植难度，当前 UB 模型不评分 atomic |
| [engram_grad_w_reduce_kernel.py](../../../../TileKernels/tile_kernels/engram/engram_grad_w_reduce_kernel.py) | 外层串行选行，内层 lane 沿 j 连续；不是相邻 lane 跨行归约。不能把串行行访问当成 warp bank 冲突 |
| [engram_gate_kernel.py](../../../../TileKernels/tile_kernels/engram/engram_gate_kernel.py) backward | `dldg_smem` 由每 warp lane0 写入，再按 head 广播读取；多数大 shared 访问沿 hidden 连续。跨 warp 不等于同请求内冲突 |
| [mhc/sinkhorn_kernel.py](../../../../TileKernels/tile_kernels/mhc/sinkhorn_kernel.py) | backward 确有实际迭代的 shared 中间量 xs/sums，比 forward 的纯 fragment 更值得保留观察；但完整 fragment↔shared copy 未给出稀疏 lane 证据。不能仅凭 repeat=10 推断 padding 有收益 |
| [reduce_fused_kernel.py](../../../../TileKernels/tile_kernels/moe/reduce_fused_kernel.py)、[expand_to_fused_kernel.py](../../../../TileKernels/tile_kernels/moe/expand_to_fused_kernel.py) | 不规则索引主要访问 GM，核心累加/复用在 fragment/local，源码没有目标 shared 矩阵。不是当前 UB padding 的直接对象 |

结论：找到了更合适的真实算法候选，优先级是独立 transpose → per-channel
cast+transpose → SwiGLU 融合 transpose；目前仍没有新增的 Ascend 实测收益证明。
