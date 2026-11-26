# 完整转置与短轴 padding 实验

两条实验都沿用 `baseline / ownership / padding / auto`：输入 DSL IR 相同，仅切换
ownership/padding pass 开关。保存输入结构 hash、推导/物理 lowering/最终分配 IR、
Ascend C 源码、候选明细、数值验证和端到端 kernel 耗时。
没有选中 padding 是有效结果，不会报告成 padding 加速。

## 运行

在配置好 CANN、torch_npu 和本仓 TileLang 编译库的设备环境中，从 tilelang 根目录运行。
先查看计划，不需要 NPU，也不会编译或执行 kernel：

```bash
python examples/ascend/layout_workloads/run_padding_study.py --dry-run --output /tmp/padding-study
```

运行两个方向，默认共 120 个 case/mode 作业：

```bash
python examples/ascend/layout_workloads/run_padding_study.py --bench --output /tmp/padding-study
```

也可以分开运行：

```bash
python examples/ascend/layout_workloads/run_padding_study.py --study transpose --bench --output /tmp/transpose-study
python examples/ascend/layout_workloads/run_padding_study.py --study short-axis --bench --output /tmp/short-axis-study
```

先做单个矩形 tile 的四模式正确性 smoke check：

```bash
python examples/ascend/layout_workloads/run_workloads.py --case transpose --rows 64 --columns 128 --threads 32 --blocks 1 --output /tmp/transpose-smoke
```

补汇总必须使用原运行的相同参数（包括 `--bench`、线程数、扫描类型等），不重新执行：

```bash
python examples/ascend/layout_workloads/run_padding_study.py --bench --summarize-only --output /tmp/padding-study
```

默认每个 tile/模式使用 32 blocks、32 threads；设置 `--blocks`、`--threads` 可改变，
参数必须满足各扫描形状的约束。每个模式独立进程运行，日志保存在对应
`CASE/MODE.worker.log`。某个作业失败后仍继续其他作业，最终以非零状态退出并保留已有报告。
`--rounds` 控制独立计时批次；每批仍使用 runner 的 warmup/repeat。

## 方向一：完整 FP32 转置

新增 `transpose` case，测试 `64x64 / 64x128 / 128x64 / 128x128` 四种 tile。
输入 `[blocks, rows, columns]`，输出 `[blocks, columns, rows]`，逐元素精确比较 CPU 转置。

```text
GM 输入 → 紧密 source UB
        → SIMT：transposed[c,r] = source[r,c]
        → GM 转置输出
```

所有元素都参与，不只提取一段列写回，不增加人工重复。端到端计时包含输入 DMA、
SIMT 转置、必要同步和输出 DMA；模型 entry 只解释其中的 SIMT UB 部分。

算法来源为工作区 [TileKernels batched transpose](../../../../TileKernels/tile_kernels/transpose/batched_transpose_kernel.py)。
这是原生 Ascend 的完整算法改写，**不是**原 CUDA 寄存器 4x4 微块和行旋转调度的等性能移植。
原源码已有 `+4 elements` padding 和 rotation；不能把本例无 padding 基线称为原库默认性能。

除四个自动模式外，还运行两个单独标记的手工参考：

| 参考 | 实际实现 | 比较边界 |
| --- | --- | --- |
| `row32` | 转置目标逻辑 Buffer 每行加 32B，直接步长 DMA | 显式物理布局，关闭两个自动优化开关 |
| `source4_repack` | 转置先写入 `(columns, rows+4)`，再用另一个已调度 SIMT VF 搬到紧密 UB，最后 DMA | 保留原源码 +4 FP32 元素的行跨度；计入额外 UB、VF 和完整 repack 的时间，不复现 rotation |

FP32 的 +4 元素只增加 16B，不能直接假设满足当前 UB DMA 的 32B pitch 对齐要求。
因此该参考不直接 DMA 未对齐布局，也不把 repack 成本移出计时。仅比较同一批输入和
相同转置输出的完整耗时；其 IR 与四模式自动实验明确不同。

需要单跑手工参考时使用独立目录：

```bash
python examples/ascend/layout_workloads/run_workloads.py --case transpose --mode baseline --transpose-layout row32 --rows 64 --columns 128 --bench --output /tmp/transpose-row32
python examples/ascend/layout_workloads/run_workloads.py --case transpose --mode baseline --transpose-layout source4_repack --rows 64 --columns 128 --bench --output /tmp/transpose-source4
```

## 方向二：短轴形状扫描

复用 `gemm_scale`、`attention_finalize`，不将其称为完整 GEMM 或 Attention。
两者都保留二维 UB 和真实运算，四模式没有人为固定一个较差的线程划分。

| 扫描 | 形状 | 控制的变量 |
| --- | --- | --- |
| `fixed-rows` | 默认 `128 x {4,8,16,32,34,64}` | 固定行数；工作量随列数变化，34 是非二次幂控制组 |
| `equal-elements` | `1024x4,512x8,256x16,128x32,64x64,32x128` | 每 tile 均为 4096 元素，便于区分形状与工作量影响 |

前者可用 `--rows` 调整，后者保持 4096 元素。`--short-scan` 可选择其中一组。
如需同时观察稀疏单列写回，可使用：

```bash
python examples/ascend/layout_workloads/run_padding_study.py --study short-axis --short-cases gemm_scale attention_finalize kda_column_store --bench --output /tmp/short-axis-with-kda
```

不要预设短轴一定受益：短行可能本就分散到不同 group，padding 也可能被 DMA
对齐约束过滤。单列 case 的有效更新量仅为 `rows*blocks`，并非整个矩阵；完整搬运
开销可能掩盖一次列写回的改善。

## 结果怎么看

顶层 `summary.md / summary.csv` 汇总两个方向；`study_plan.json` 保存完整实验参数。
同时生成 `cost_summary.md / cost_summary.csv`：Markdown 按扫描、形状和手工布局分表，
列出 case、iterations、mode、padding scheme、最终 IR UB 占用及相对 baseline 的增量、
模型 UB latency、instructions、requests、entries，以及实测 kernel time（us）和加速比。
模型成本与端到端实测时间分别列出；没有计时的数据保留 unknown。
已有逐模式 JSON 可用上面的 `--summarize-only` 命令补生成，无需重跑设备实验。
每个子配置还保留原 runner 的 `summary.md / summary.csv / candidates.csv` 和逐模式 JSON。

重点顺序：

1. 先确认正确性通过，四个自动模式的输入 IR hash 一致，且配置、源码 hash 相同。
2. 检查 `padding_scheme`、`padding_applied`、真实 loop layout，确定实际采用了什么。
3. 比较 `entries / estimated_latency_cycles / cost_scope`，确认收益不是遗漏了访问。
4. 检查最终 UB 分配，以及 `dma_audit.descriptors` 的 `burst_count / burst_bytes /
   gm_pitch_bytes / ub_pitch_bytes`。它们是静态描述符字段，不是动态 DMA 指令次数或耗时。
5. `ub_store_audits` 给出 scalar lowering 的活跃 lane 和 group/line 样本；出现 vectorized
   或无法求值时保留 unknown，不能把未知当作无冲突。读取与最终 b32/b64 指令仍需查看
   `.lowered.tir`、`.cpp` 和设备反汇编。
6. 最后比较同配置端到端时延及原始样本，分别判断 padding-only、ownership-only 和 auto。
   手工参考与 native baseline 比完整耗时，不能声称是同 IR 的 pass 开关比较。

自动顺序是先 ownership 再 padding；auto 不一定优于 padding-only。若 ownership
先改变访问模式，使后续候选不同，应该作为顺序搜索的取舍报告，不应删掉较差结果。

## 当前验证范围

本地已运行主机侧数值参考、扫描和报告测试，并对代表性转置/短轴配置执行布局推导与
物理 lowering。转置的自动 padding、row32、source4_repack 三条路径已完成完整
Ascend lowering 和 C++ 源码生成（未调用设备编译器）。开发机没有 CANN 和 NPU，
尚无新增设备正确性、反汇编或实测加速结果。
`source4_repack` 也不构成已完成原 CUDA 微块调度移植的证明。

本地编译库的地址模型预筛选结果如下（32 threads、每 block 一个 tile）：

| case | tile | baseline entries | ownership entries | padding entries | auto entries | padding-only 选择 |
| --- | --- | --- | --- | --- | --- | --- |
| transpose | 64x64 | 4224 | 2048 | 1152 | 2048 | transposed 每 8 行 +32B |
| transpose | 64x128 | 8448 | 6144 | 2304 | 6144 | transposed 每 8 行 +32B |
| transpose | 128x64 | 8448 | 4096 | 2304 | 4096 | transposed 每 8 行 +32B |
| transpose | 128x128 | 16896 | 12288 | 4608 | 12288 | transposed 每 8 行 +32B |
| gemm_scale | 128x4 | 128 | 128 | 128 | 128 | 未选择 |
| gemm_scale | 128x8 | 512 | 256 | 512 | 256 | 未选择 |
| gemm_scale | 128x16 | 2048 | 256 | 1024 | 256 | work 每 16 行 +32B |
| gemm_scale | 128x32 | 8192 | 256 | 2048 | 256 | work 每 8 行 +32B |
| gemm_scale | 128x34 | 544 | 544 | 544 | 544 | 未选择 |

这些是模型结构计数，**不是 ISA 统计或实测加速**。短轴本身并不保证 padding
有优势：列宽 4、8 的例子没有选择 padding，16/32 列则 ownership 模型结果更好。
转置中 padding-only 的 entry 更少，而 auto 的顺序搜索保留了另一套 ownership，
是值得上板进一步分析的取舍。每条路径的 DMA、同步和真实指令开销仍需计时验证。
