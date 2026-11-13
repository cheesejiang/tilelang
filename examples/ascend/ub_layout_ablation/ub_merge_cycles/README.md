# SIMT UB 8B 读合并的 cycles 对照实验

目标：检查同一个 warp 内，不同 lane 读取同一个 8B 对齐块内的两个
不同 float 地址，是否具有与可并行访问接近、且低于资源冲突访问的耗时。
本实验使用原生 SIMT，不调用 TileLang 成本模型。

适用环境：Ascend 950PR/950DT（dav-3510）、CANN >= 9.1.0、ASC CMake 包、
可用设备 0。先加载目标机 CANN 的 set_env.sh，在 tilelang 仓库根目录运行：

```bash
python3 examples/ascend/ub_layout_ablation/ub_merge_cycles/run.py --output /tmp/ub_merge_cycles
```

脚本为每个 case 编译独立程序，分别测量 100、1000 次 UB load + float add。
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

所有 case 的输入值均非零，使用 volatile UB 指针保留重复读取。
计时区域与上游样例一致：clock() 包围重复读取及累加；UB 初始化在前、
输出写回在后。保留上游 CANN Open Software License 声明。
来源：工作区 asc-devkit 的 examples/03_simt_api/02_features/01_api_features/
00_memory_access/bank_conflict。

## 采集和判断

结果包含各次运行日志、32 个 lane 的原始 cycles、build.log 和 summary.csv。
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
   `#pragma unroll` 和 `volatile` 是源码意图，最终是否满足仍以指令为准。
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

## 当前验证状态

已构建测试源码和采样脚本；当前开发环境没有 CANN/ASC 和 NPU，尚未编译设备
程序或采集真实 cycles。文档不填入上游参考值作为本测试的结果。
