# SIMT UB 8B 读合并的 cycles 对照实验

目标：检查同一个 warp 内，不同 lane 读取同一个 8B 对齐块内的两个
不同 float 地址，是否具有与可并行访问接近、且低于资源冲突访问的耗时。
本实验使用原生 SIMT，不调用 TileLang 成本模型。

适用环境：Ascend 950PR/950DT（dav-3510）、CANN >= 9.1.0、ASC CMake 包、
可用设备 0。先加载目标机 CANN 的 set_env.sh，在 tilelang 仓库根目录运行：

```bash
python3 examples/ascend/ub_layout_ablation/ub_merge_cycles/run.py --output ./tmp/ub_merge_cycles
```

脚本为每个 case 编译独立程序，分别测量 100、1000 次 UB load + 累加。
每组舍弃前 2 次进程运行，采集后续 11 次。每次运行只启动一个 block、32 个
线程，并校验所有线程结果；预热进程仅用于设备稳定，不保留同一 kernel 实例状态。
任一编译、执行或数值检查失败即停止。

| case | 每个 lane 的 UB 字节偏移（t=lane id） | 对照意义 |
| --- | --- | --- |
| 0 | 0 | 同地址广播/合并 |
| 1 | 8*t | 分散资源的基线 |
| 2 | 4*t | 连续 float，每对相邻 lane 共享一个 8B 块 |
| 3 | 512*t | 同 bank/subbank 不同行冲突 |
| 4 | 256*t | 同 bank group、同 subbank 冲突 |
| 5 | 4*(t%2) | 核心实验：两个不同地址在同一个 8B 块内 |
| 6 | 8*(t%2) | 两个不同 8B 块、不同资源 |
| 7 | 256*(t%2) | 两个不同 8B 块、相同竞争资源 |
| 8 | 0 | uint64_t，同地址读完整 8B |
| 9 | 8*t | uint64_t，连续不重叠地读完整 8B |
| 10 | 128*(t%2) | float，同 virtual group、不同 key/line 的双地址 |
| 11 | 128*t | float，32 个不同 line 集中到一个 virtual group |
| 12 | 512*(t%2) | float，同 key 但不同完整 line 的双地址 |
| 13 | 16*t | float，只使用 8 个 virtual group 的分布 |

case 0～7、10～13 使用 `float`，每 lane 每轮读取 4B；case 8～9 使用 `uint64_t`，
每 lane 每轮读取 8B，并进行 uint64_t 累加。case 9 的“连续”指元素起始地址
相差 8B，不是以 4B 步长读取重叠的 8B 范围。8B 输入的高低 32 位均有非零
数据，输出按完整 64 位累加结果精确校验。

所有 case 的输入值均非零，使用不带 `volatile` 的普通 UB 指针，以观察正常优化下的指令选择。
重复读取的地址不变，编译器可能将读取外提、合并或消除；必须检查最终设备代码中的
动态读取次数，不能直接将 cycles/N 当作每次 UB 读取成本。旧版带 `volatile` 的结果
应单独保存，不与当前版本直接混合比较；去掉该修饰符不保证一定生成 LDS。
计时区域与上游样例一致：clock() 包围重复读取及累加；UB 初始化在前、
输出写回在后。保留上游 CANN Open Software License 声明。
来源：工作区 asc-devkit 的 examples/03_simt_api/02_features/01_api_features/
00_memory_access/bank_conflict。

## 采集和判断

结果包含各次运行日志、32 个 lane 的原始 cycles、build.log、summary.csv 和
summary.md。CSV 和 Markdown 表格包含实验名 `name`、读取类型 `dtype`、
每 lane 字节数 `bytes_per_lane`、迭代数、样本数、最小/中位/最大 cycles 和
每次迭代平均 cycles；程序结束时也会打印完整 Markdown 表格。
每次运行以 32 个 lane 的最大 cycles 作为一个 warp 样本，再对运行间取中位数。
32 个 lane 不是 32 次独立实验。cycles/iteration 包含 load、累加依赖与计时开销，
不等于单条 LDS 延迟；总 cycles 不应再乘以 32。

先检查编译后的 ISA：计时区间应为相同数量和宽度的 UB load 及累加，
不能有省略读取、不同访存宽度或额外 spill。构建目录保留以便反汇编检查；
脚本不自动认定 ISA 等价。

如果 case 5 接近 case 6，且 case 7 稳定更慢，同时 case 3/4 的明显冲突
得到复现，则结果支持“同一 8B 块内的不同地址可合并服务”。case 0 快只能
支持同地址广播，不能单独支持不同地址的 8B 合并。两种重复次数应有一致趋势；
还可通过两点 cycles 差除以迭代数差，减弱固定计时开销影响。
case 7 只有两个唯一地址，差异可能小；若样本波动覆盖差异，结果应记为
证据不足，增加采样、改变访问模式并检查 ISA，不能宣称验证成功。

该计时实验是行为证据，不直接观察硬件内部的合并事件，也不能验证成本模型
的所有 entry 打包、写入和周期参数。尤其不测试多个 lane 写同一地址。

## 自动提取与反汇编

在目标设备加载 CANN 环境后，运行：

```bash
python3 examples/ascend/ub_layout_ablation/ub_merge_cycles/run.py --disassemble --output /tmp/ub_merge_cycles
```

可通过 `--msobjdump /实际路径/msobjdump --objdump /实际路径/llvm-objdump`
指定与 CANN 配套、支持 dav-3510 的工具。脚本使用参数数组执行工具，不使用 shell
拼接。默认仅测 cycles；`--disassemble` 才启用设备代码提取及解码。

已有测量产物可以补做反汇编，不重新编译、也不重新测量。下面指定已有的 case
范围；旧版本没有 case 10～13，不能对其直接使用所有 case 的默认值：

```bash
python3 examples/ascend/ub_layout_ablation/ub_merge_cycles/run.py --disasm-only --cases 0 1 2 3 4 5 6 7 8 9 --output /tmp/ub_merge_cycles
```

只测新增 entry 对照也可以选择子集：

```bash
python3 examples/ascend/ub_layout_ablation/ub_merge_cycles/run.py --cases 0 1 5 6 7 10 11 12 13 --disassemble --output /tmp/ub_merge_entry_checks
```

每个 `build/caseX_nY/disasm/` 包含工具信息、meta、ELF 列表、提取日志、设备 ELF、
`.info.txt`、`.asm`、关键词导航 `.hints.txt`、`manifest.json` 和人工核对模板
`review.md`。提取到的多个 ELF 分别处理，常见 host CPU ELF 会被跳过。
`measurement.json` 记录测量二进制的 SHA256；有该记录时，补做反汇编会检查
二进制是否匹配。旧产物没有哈希时，来源关联标为 unknown，不能声称已匹配。

`summary.csv` / `summary.md` 增加反汇编状态及 `model_*` 预测列。
`output_generated_unreviewed` 仅表示生成了解码输出，**不表示已确认 b64 或 ISA
等价**；必须完成 `review.md`。工具缺失、解码失败或出现 `<unknown>` 等情况
会记录失败并在最终返回非零退出码；已采集的 cycles 仍会保存。
`--disasm-only` 独立输出 `disassembly-summary.json`，不改写历史测量表。

## 提取设备代码与反汇编

下面的命令在完成前面的运行后执行，使用同一批实际测量过的二进制，
不重新编译或改变优化选项。以 case 5、100 次迭代为例：

```bash
ub_case_build=/tmp/ub_merge_cycles/build/case5_n100
mkdir -p "$ub_case_build/disasm"
msobjdump -h
msobjdump --dump-elf "$ub_case_build/demo" --verbose > "$ub_case_build/disasm/meta.txt"
msobjdump --list-elf "$ub_case_build/demo" > "$ub_case_build/disasm/elf-list.txt"
msobjdump --extract-elf "$ub_case_build/demo" --out-dir "$ub_case_build/disasm"
cat "$ub_case_build/disasm/elf-list.txt"
```

`msobjdump` 随 CANN 提供；未进入 PATH 时，在当前 CANN 安装目录的
`tools/msobjdump` 下查找，按实际安装布局调用。它在这里负责 ELF 提取及元信息
解析，`--dump-elf` 本身不输出汇编指令。

根据 `elf-list.txt` 和 `meta.txt` 选择包含 `bank_conflict_kernel` 的设备 ELF。
直调样例的文件名可能类似 `demo.aicore.o`，但必须以实际输出为准。
一个产物可能包含多个设备 ELF，不能随意选择第一个，也不要直接对 host 的
`demo` 使用系统 objdump：那通常看到的是 CPU 启动代码。

加载 CANN 环境后，检查当前工具链提供的 `llvm-objdump`：

```bash
command -v bisheng
command -v llvm-objdump
llvm-objdump --version
llvm-objdump --help
```

应使用与当前 CANN/Bisheng 配套且支持 dav-3510 设备代码的工具，避免误用
系统 LLVM。若未进入 PATH，可在 CANN 编译器安装目录查找并用绝对路径调用。
下面用 `demo.aicore.o` 举例，先把路径改成上一步实际提取的设备文件名：

```bash
ub_device_elf="$ub_case_build/disasm/demo.aicore.o"
llvm-objdump --file-headers --section-headers --syms "$ub_device_elf" > "$ub_case_build/disasm/device-info.txt"
llvm-objdump -d --demangle "$ub_device_elf" > "$ub_case_build/disasm/device.asm"
rg -n -i 'bank_conflict|lds|sts|clock|timestamp' "$ub_case_build/disasm/device.asm"
```

这些是标准 LLVM objdump 用法，不保证每个 CANN 发行包都开放 dav-3510 SIMT
反汇编支持。出现“不支持目标架构”、大量 `<unknown>` 或只有原始字节时，
不能算反汇编成功；应检查该版本配套工具的帮助/文档，使用其支持的设备反汇编
工具。不要靠强制指定一个不匹配的 CPU 架构消除报错。当前开发机没有该工具链，
以上设备提取和解码命令尚未本地实测。

如果目标版本暂时不能解码，保留原始 `demo`、设备 ELF、工具版本及报错。
编译器生成的 `.s` 可以辅助理解，但重新编译得到的汇编不能替代对已测量二进制的
核对；LLVM IR 也不是最终机器指令。此时结论应标注“尚未核对最终指令”。

提取命令依据工作区中的 [CANN msobjdump 文档](../../../../../asc-devkit/docs/zh/guide/programming_guide/appendix/msobjdump_tool.md)。

## 如何分析反汇编结果

先比较相同迭代数的 case 5、6、7，再检查 case 0～4；100 和 1000 次版本分别核对。
对每个构建目录重复上面的提取步骤，保留独立的 `disasm` 目录。

1. **定位正确函数和计时边界。** 找到 `bank_conflict_kernel` 的设备实现；
   如果入口调用 SIMT 子函数，继续跟踪到真正执行读取的子函数。定位两次
   `clock()` 对应的计数器读取，确认 UB 读取及累加处于它们之间，
   初始化和输出位于区间外。具体助记符取决于工具链，搜索词只是导航，
   不保证指令名含有 `clock`。还要检查结束计时与累加结果的依赖/等待关系；
   不要仅凭源码顺序认定所有访存都已完成，必要时参照该版本指令说明。
2. **核对动态读取次数。** 每轮源码应保留一次每 lane 的 float UB 读取。
   完全展开时，计时区间应有 N 次对应的读取；若保留循环，则按“循环体内
   读取数 × 实际循环次数 + 尾部读取数”计算。不能只用全文 `rg -c lds`
   当作动态指令数，因为它可能包含初始化、别的函数，或忽略循环次数。
   当前不使用 `volatile`，`#pragma unroll` 也不保证每轮实际执行读取。
   如果读取被移到计时区间外，或动态次数少于 N，该结果不能用于验证 N 次重复访问的合并成本。
3. **核对访问类型、宽度和活跃线程。** 检查是 UB load（可能显示为 LDS 类
   指令），每 lane 读 4B，且 32 个 lane 的执行掩码一致。b32 是 32 bit，
   即每 lane 4B；不要误读为每 lane 32B。若某组变成更宽的读取、其他存储空间
   的读取，或只让部分 lane 执行，则该组与其他组不能直接归因对比。
4. **追踪地址，而不是只比指令文字。** 从 lane id 的取余/位运算、移位和
   UB 基址加法中还原地址：case 5 为 `base + 4*(t%2)`，case 6 为
   `base + 8*(t%2)`，case 7 为 `base + 256*(t%2)`。还要确认 `base`
   至少 8B 对齐，使 case 5 的两地址确实落在同一 8B 块。不同立即数、寄存器
   编号或计时区间外的地址准备差异是预期现象，不要求整份汇编完全相同。
5. **检查其他开销。** 比较计时区间的浮点加法、依赖链、分支、等待/同步、
   地址计算和额外访存；关注 spill/reload（寄存器溢出及重载）。如果某组有额外
   开销，先解释它对 cycles 的影响。不要为让文本相同而修改编译选项后仍引用
   旧 cycles；任何重新编译都需要重新测量并配套保留二进制。

下面只是阅读思路示意，**不是真实 dav-3510 汇编或精确指令语法**：

```text
start = read_cycle_counter
重复 N 次:
    value = UB_LOAD_4B(base + lane_offset)
    acc = FLOAT_ADD(acc, value)
end = read_cycle_counter
```

三组都应该保留相同的 N 次读取和累加，主要区别为 `lane_offset`。
硬件内部合并不要求出现一条单独的“merge 指令”，也不能从一条 LDS 的反汇编
直接读出实际发布了几个 entry。

建议将核对结果和 cycles 放在一起记录，未核对项填写 unknown：

| case / N | 设备 ELF / 函数 | 计时区间动态 UB load 数 | 每 lane 宽度 / 活跃 lane | 地址与对齐正确 | 额外开销 | median cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 5 / 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待测 |
| 6 / 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待测 |
| 7 / 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待测 |

只有在读取宽度、次数、掩码和主要执行开销可比后，才用 cycles 判断地址模式的
影响。如果 case 5 接近 case 6，case 7 稳定更慢，可支持“同一 8B 块内不同
地址合并服务”的解释；如果 case 5 同样慢，或三组差异小于波动，应记录为
不支持/证据不足，再检查指令和实验设计。反汇编本身不能证明硬件完成合并。

## 新增 8B 读取结果如何判断

先在相同类型内比较 case 8（同地址）和 case 9（连续地址），结合已有 case 0
和 case 2 的 float 结果展示 4B/8B 的区别。case 8/9 必须额外检查：最终读取
到底是一条 b64，还是两条 b32；高低两半数据是否都被读取并参与结果计算。
不能仅凭源码 `uint64_t` 就认定最终发射了一条 b64。分析表格时，应将已确认的
实际指令宽度单独记录。

8B 组使用整数累加，4B 组使用浮点累加；因此跨类型的耗时差还包含累加指令、
依赖和可能不同的寄存器开销。case 8 与 case 9 的对照可以描述完整 8B 读取的
行为，但即使同地址 b64 很快，也不能单独证明两个内部 request 被硬件跨请求
合并：还可能存在流水重叠。此实验不直接观察硬件内部 request 边界。

## 要验证的两条规则与证据边界

### 每个 lane 的 4B 与内部 request

“一个 lane 只能读取 4B”不是本实验预设的硬件结论。应区分：

- 指令：b32 每 lane 读 4B；如果 case 8/9 解码为 b64，则每 lane 一条指令读 8B。
- 当前成本模型的 request：每 lane 至多 4B；b64 按两个独立的 4B slice 计价。
- 反汇编可观察指令宽度和次数，不能直接观察 b64 内部拆成几个 request。

在 case 8/9 的 `review.md` 中记录实际是“一条 b64”还是“两条 b32”，
并确认两半数据都参与最终结果。即使观察到一条 b64，也不能由此确认内部
两个 request 是否独立合并。表中 `model_requests_per_iteration` 是模型假设，
不是反汇编或硬件测出的 request 数。

### 为什么按 virtual group 和完整 line 分 entry

本实验按以下待验证规则计算预测，假定 UB 基址满足相应对齐：

```text
line(addr) = addr & ~7
key(addr) = (addr >> 3) & 63
virtual_group(addr) = key(addr) & 15
```

一个 entry 对每个 virtual group 只能容纳一个完整 line。同 line 可以合并；
同 group 的不同 line 要分 entry。关键对照及预测如下，均为每轮访存：

| case | 访问偏移 | dtype / 每 lane | 模型 request 数 | 模型 entry 总数 | 要区分的条件 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | float / 4B | 1 | 1 | 重复地址仍占 64 个 fragment |
| 5 | 0 / 4 | float / 4B | 1 | 1 | 同完整 8B line |
| 6 | 0 / 8 | float / 4B | 1 | 1 | 不同 virtual group |
| 10 | 0 / 128 | float / 4B | 1 | 2 | 同 virtual group，不同 key/line |
| 7 | 0 / 256 | float / 4B | 1 | 2 | 同 virtual group，不同 key/line |
| 12 | 0 / 512 | float / 4B | 1 | 2 | 同 key，但不同完整 line |
| 1 | 8*t | float / 4B | 1 | 2 | 32 个 line 分布到 16 个 virtual group |
| 13 | 16*t | float / 4B | 1 | 4 | 32 个 line 分布到 8 个 virtual group |
| 11 | 128*t | float / 4B | 1 | 32 | 32 个 line 集中到一个 virtual group |
| 3 / 4 | 512*t / 256*t | float / 4B | 1 | 32 | 已有密集冲突对照 |
| 8 | 0，每 lane 读 8B | uint64_t / 8B | 2 | 2 | 两个 slice 分别打包，不跨 request 合并 |
| 9 | 8*t，每 lane 读 8B | uint64_t / 8B | 2 | 4 | 每个 slice 各需两个 entry |

`model_entries_per_iteration` 是上述规则的解析预测，不调用 C++ 模型，也不
参与 cycles 测量。当前对齐输入每 request 共 64 个 fragment，容量约束不会
在 group/line 约束之外再增加 entry，因此可用“同一 group 的不同 line 数的
最大值”计算每 request 的预测。该快捷计算不适用于任意未对齐访问。

尤其比较 case 10 与 case 6：按 bank group/subbank 的文档资源划分，0 和
128B 可以落在不同竞争资源，而按这里的 **16 个 virtual group** 则必须拆开。
因此 virtual group 不是物理 bank group 的别名。这个对照用于区分规则，
不能只看 0/256B 或 0/512B 冲突后就认定 virtual_groups=16 得到验证。
若 case 10 与 case 6 相同，也不能仅凭 cycles 相同就彻底否定拆 entry：额外
发布可能被其他执行开销隐藏。应结合 case 1、13、11 的集中程度变化看趋势。
反之若明显变慢，也需要先排除其他指令和资源瓶颈。

### 64 个 2B fragment，重复地址计数

模型对每个活跃 lane 的 4B 读取计两个 2B fragment，重复地址也不去重：

```text
32 lanes * (4B / 2B) = 64 fragments / request
```

同地址 float 读取虽然只涉及一个唯一 4B 地址，模型仍记 64 个 fragment；
每 lane 读 8B 的模型为两个 request，每个 64 个，而不是把 128 个塞进一个。

**这些测试不能独立验证“entry 上限恰为 64，重复地址也消耗容量”。**
原因是全部对齐、单 warp、每 lane 至多 4B 的 request 最多只有 64 个 fragment，
无法构造 65 个 fragment 触发边界。把 lane 数增加到另一个 warp，或用两条
指令凑够 65 个，也不再是同一个 request；不能据此推断 entry 容量。
即使模型统计正确，也只能证明实现按规则计数，不能证明硬件容量。
本次将该项明确保留为“硬件未验证”，需要硬件资料或能观测内部请求/entry 的
额外手段；计时与普通反汇编都无法直接给出该容量。

最终记录应分别给出“指令宽度已核对”“group/line 行为支持或证据不足”以及
“内部 request 和 64-fragment 容量尚未验证”，不能把它们统一写成全部验证通过。

## 当前验证状态

case 0～7 的 100/1000 次结果已由用户在目标环境采集并提供；这些结果支持读合并
和冲突趋势。新增 case 8～13 及自动反汇编尚未在当前开发环境执行：本机没有
CANN/ASC 和 NPU。工具链解码能力、真实指令宽度和 entry 假设仍待目标机核对。
模型预测列不是新增硬件测量结果。
