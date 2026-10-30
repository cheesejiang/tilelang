# UB ownership / padding 四状态对比

固定每次插入 **32B**，改变插入周期。自动搜索相邻 lane 元素差的
1～32 倍周期，仅保留完整行周期；先比较整个组件的 SIMT UB 访问成本，
成本相同时选择未按分配对齐取整的 UB 字节数更小的方案。
不 padding 是基线，同成本不会增加存储。多个 buffer 按顺序改进，
这是有限候选搜索，不保证任意访问或多个 buffer 的联合全局最优。
DMA 成本目前不计入评分，但每个候选必须通过 DMA 合法性检查。

直接 DMA 示例的显式 padding 为每 `8 * row_stride` 行加 32B：
`row_stride=1` 时每 8 行，`row_stride=16` 时每 128 行。
32 个 lane 的规则 f32 跨行读取预计从 32 entry 降到 8 entry。
这些预期仍需在设备上编译并验证正确性与整 kernel 性能。

## 直接验证 DMA + 自动 padding（优先运行）

`example_dma_padding.py` 直接编译原始 DSL kernel：GM→UB、规则跨行读取、UB→GM。
没有分析 kernel、布局重放、循环平移或运行时 guard。
`auto` 模式在正常编译流水线中选择布局，并验证设备输出。未选择 padding 时
明确报告 `padding NOT APPLIED`，不能把该结果当作 padding 收益验证；
使用 `--require-auto-padding` 可将此情况作为回归失败。baseline 固定线程布局且不加 padding；ownership 用地址等价
的一维 UB 阻止 padding 搜索；padding 显式注解物理布局并固定线程布局。
跨行访问每线程只有一个输出，ownership 模式可能与 baseline 相同。

```bash
python examples/ascend/ub_layout_ablation/example_dma_padding.py --mode auto
python examples/ascend/ub_layout_ablation/example_dma_padding.py --mode auto --require-auto-padding
python examples/ascend/ub_layout_ablation/example_dma_padding.py --row-stride 16
python examples/ascend/ub_layout_ablation/example_dma_padding.py --row-stride 1 --bench
```

目前新 DMA lowering 仅支持：完整静态 UB tile、同 dtype、连续且可证明不越界的
GM 区域、完整 padding 周期，不与 `pad_value/data_select` 尾部填充组合。
自动候选暂不覆盖带流水线/显式多版本注解的 kernel，避免后续 ring 轴物化改变此契约。
每个周期是一个 DMA burst，GM pitch=K，UB pitch=K+p，单位换算为字节。
UB padding 空洞不被搬运，也不被写回 GM。每行 padding 是 K=一行元素数的特例。
部分区域/OOB、非连续 GM 或不匹配的 layout 不作为自动 padding 候选。
它们不应靠仅更改起始指针来隐式获得支持。

完整编译仍会经过 AutoSchedule/InsertSync。此前的 `IterSumExpr` 是另一项已知问题，
本次 DMA 支持并未修复该问题；如果这里也触发，应保留这个直接例子的完整报错。
本地没有 TVM 编译库和 NPU，尚未执行此设备验证。
每个模式先写入 `<mode>.tir`、`<mode>.cpp` 和 `<mode>.json`，再检查布局结果；
JSON 包含实际 buffer 名称、逻辑/物理形状和布局。即使断言失败也保留诊断文件。

## 原布局重放实验

本目录使用 Ascend DSL，比对三个规则访问场景的四种布局状态，不修改 compiler pass：

| 模式 | 线程分配 | UB 物理布局 |
| --- | --- | --- |
| `baseline` | 显式固定，禁止 ownership 搜索 | 原始连续存储 |
| `ownership` | 使用现有 ownership 搜索 | 地址等价的一维 UB，当前 padding 搜索不处理一维 buffer |
| `padding` | 与 baseline 相同，禁止 ownership 搜索 | 跨行访问手动设置周期 padding；连续访问保持原布局 |
| `auto` | 使用当前流程的选择 | 使用当前流程的选择，可能保留原布局 |

这里“无 ownership”指**不搜索/优化 ownership**；SIMT 循环始终需要线程分配。
baseline 是可复现的固定分配，不声称等于所有形状下编译器的 native 最优方案：
每线程有足够工作时取 width=2，否则 width=1。脚本检查固定分配没有被改变。

## 场景

全部使用 f32；每个参与计算的逻辑元素执行相同的递推 `x = x * 1.0001 + 0.001`。
每轮有两个更新阶段：第二阶段把迭代编号循环平移一位，使 baseline 的相邻线程
交接相同元素，通过现有同步分析保留 UB 数据依赖，减少整个递推被直接提升到
线程私有寄存器的机会。每个元素共更新 `2 * iterations` 次，两个阶段都被计时。
第二阶段只有规则步长和末尾循环回绕；padding 候选由无回绕的第一阶段生成，
评分包括两个阶段。是否最终保留目标访存仍需要检查实际生成代码。

| case | UB 逻辑形状 | 参与递推的元素 | padding-only |
| --- | --- | --- | --- |
| `contiguous` | 8192 个元素 | 全部元素，128 个线程 | 无 padding；用于比较 ownership |
| `row1` | 32×32 | 每行第 0 列，32 个线程 | 每 8 行插入 32B |
| `row16` | 512×32 | 每 16 行的第 0 列，32 个线程 | 每 128 行插入 32B |

跨行场景 baseline 的每 lane 一个元素，初始化仍覆盖完整 UB。
周期布局为 `F(i) = i + floor(i / K) * p`，K 是周期元素数，p 是 padding 元素数。
`row1` 使用 K=32、p=2，`row16` 使用 K=512、p=2。
不在最后一个元素之后额外分配无用尾部空洞。

## 为什么分分析和测量两个 kernel

最初编写本实验时，自动 padding 会排除含 GM↔UB DMA 的组件。
当前已支持上面所述的完整周期 DMA；下面保留原布局重放实验，用于隔离
连续访问的 ownership 和更复杂的重复访问行为，而不将它当作直接 DMA 验证。

所以本例先对纯 UB 的 DSL kernel 执行布局推导所需的编译前缀，停在
`tl.AscendLayoutInference`，读取 loop/storage layouts；随后将选出的布局
显式重放到**相同结构**的测量 kernel：

```
GM → dense incoming UB → pack into work UB
                              ↓
                    repeated SIMT updates
                              ↓
GM ← dense outgoing UB ← gather active elements
```

work 的物理地址和初始化/计算的线程分配与分析结果保持一致。
DMA 只访问 dense staging buffer，因此不依赖尚未支持的 DMA padding lowering。
所有模式使用同一测量模板、输入、递推次数和输出契约。

这验证**被选布局的可执行性与端到端效果**，不是声称自动 pass 已能直接处理含 DMA
的原始 kernel。ownership-only 的一维表示也是本例的隔离手段，不是新增 pass 开关。

## 运行

先构建本工作区中包含 padding 改动的 Ascend 后端，并配置 CANN。
设备验证需要可用的 Ascend 950、配套 torch/torch_npu 和 Bisheng。

从仓库根目录执行：

```bash
# 12 个组合：3 个场景 × 4 个模式。只推导、检查和导出布局 IR，不执行 NPU。
python examples/ascend/ub_layout_ablation/example_layout_ablation.py --analyze-only
```
```bash
# 12 个组合：在 NPU 上做数值验证，跳过性能测量。
python examples/ascend/ub_layout_ablation/example_layout_ablation.py --no-bench
```
```bash
# 完整正确性与性能比较。
python examples/ascend/ub_layout_ablation/example_layout_ablation.py
```
```bash
# 单独比较跨 16 行，增加递推次数。
python examples/ascend/ub_layout_ablation/example_layout_ablation.py \
  --case row16 --iterations 128 --samples 5 --output /tmp/ub-row16
```
```bash
# 单个模式。
python examples/ascend/ub_layout_ablation/example_layout_ablation.py \
  --case row1 --mode auto --no-bench
```

`--analyze-only` 不执行设备代码，不主动导入 torch_npu；仍需要 TileLang Python
运行依赖和已编译的 Ascend 后端。若环境的 torch 自动加载 torch_npu，仍需配置
CANN，或在纯分析进程启动前关闭设备扩展自动加载。

## 输出与解释

默认输出目录 `ub-layout-results/<case>/<mode>/`：

- `analysis.tir`：分析 kernel 在布局推导后的 IR。
- `plan.json`：线程布局、物理布局、padding 周期、UB 字节数。
- `measurement.cpp`：实际被测 kernel 的生成源码，设备模式下导出。
- 根目录 `results.json`：正确性、各轮时延、中位时延、波动及 baseline/mode 比值。
- 根目录 `environment.json`：Python/包版本、设备名称、CANN 路径和运行参数。

计时使用与现有 Ascend 示例相同的 `do_bench(..., backend="msprof")`。
输入输出 tensor 预先分配；不计编译、参考结果生成和正确性检查。
多轮测量轮换模式顺序。`baseline/mode > 1` 表示该模式更快。
单模式运行没有 baseline 比值；任何失败会直接报错，不会显示成通过。

时延包含 DMA、pack/gather、同步和计算，**不是单次 UB 指令延迟**。
padding 增加的搬运和地址计算可能抵消 UB 冲突收益，改善不作保证。
如果差值接近测量波动，应视为尚不能确认收益。
递推循环也可能被后端优化；做归因前检查生成源码，必要时检查 ISA/profiler，
确认目标访存没有被 hoist、合并或消除。

分析模式不会执行 AutoSchedule、InsertSync 或生成 C++ 源码。
它通过只能说明布局推导及检查通过，不能代表完整编译或设备执行通过。
使用旧输出目录时，之前留下的 `analysis.cpp` 不属于当前运行生成的结果。

新增这些例子时，本地没有 TileLang 编译库和 NPU 环境，尚未运行完整 lowering
或设备测试；脚本中的正确性和布局检查需要在配置好的环境中执行。

## 多轮 UB 复用例子

`reuse_kernels.py` 用 Ascend DSL 定义多轮读写 kernel，仍比较 baseline、
ownership、显式 32B padding 和 auto。每轮先从跨行 work 读入 result，
再写回 work，最后收集输出；只在整个迭代开始/结束进行 GM 搬运。
每轮 `x = x * 0.999 + 0.001`，使结果依赖上一轮且不会很快全部收敛为同一个值。
CPU reference 按相同轮数计算。是否产生预期 UB 指令仍应检查生成的 C++/ISA，
不能仅凭 DSL 认定编译器不会提升内存访问。

| 场景 | 行宽 | 相邻 lane 跨行数 | 显式 padding |
| --- | --- | --- | --- |
| row128 | 32 个 f32 / 128B | 1 | 每 8 行加 32B |
| row64 | 16 个 f32 / 64B | 1 | 每 16 行加 32B |
| stride16 | 32 个 f32 / 128B | 16 | 每 128 行加 32B |
| rotate | 32 个 f32 / 128B | 1 | 每 8 行加 32B |

rotate 在写回时使用下一 lane 的 result，存在跨线程依赖，依赖编译器插入同步。
它衡量包含同步的迭代场景，不能当作纯 UB 带宽测试。

先做一个小配置的正确性验证，再扫描多轮性能：

```bash
python examples/ascend/ub_layout_ablation/example_reuse_suite.py --case row128 --iterations 16
python examples/ascend/ub_layout_ablation/example_reuse_suite.py --case row128 --bench
python examples/ascend/ub_layout_ablation/example_reuse_suite.py --case all --iterations 1 16 64 256 --bench
```

也可以直接运行单个配置：

```bash
python examples/ascend/ub_layout_ablation/example_dma_padding.py --case rotate --iterations 64 --bench
python examples/ascend/ub_layout_ablation/example_dma_padding.py --case reuse --columns 16 --iterations 64 --bench
```

批量脚本将每个配置放入独立进程，结果分别保存至
`ub-reuse-results/<case>/<iterations>/`。每种模式保留 `.cpp`、`.tir` 和 `.json`。
时延仍是包含 DMA、同步及调用开销的整体时延，编译和正确性比较不在计时内。
新增例子本地仅完成 Python 语法及命令行检查，尚未在 NPU 编译或执行。

批量运行还会在输出根目录生成总表：

- `summary.csv`：每行一个设置和模式，含场景、列数、跨行数、迭代次数、
  block 数、状态、是否 padding、时延（us）及相对同配置 baseline 的加速比。
- `summary.md`：每个配置一行，四种模式时延并排显示，结束时也打印到终端。

每个配置结束后立即更新总表；后续配置失败时已完成结果仍保留，失败或尚未运行
的模式会标记 `failed_or_not_run`。不带 `--bench` 时没有时延，只记录正确性状态。
每次调用重建本次扫描的总表；不会将之前运行的时延混入本次结果。

## Pass 诊断与成本总表

示例现在自动启用 `tl.ascend_layout_diagnostics`，**需要重新编译 C++ 库**。
该 PassContext 开关默认关闭，开启时在 PrimFunc 的同名属性中保存诊断记录；
不改变候选生成、比较或布局选择规则。也可以独立使用：

```python
with tvm.transform.PassContext(config={"tl.ascend_layout_diagnostics": True}):
    inferred = transform.AscendLayoutInference()(mod)
```

每个模式的 JSON 增加 `layout_diagnostics`：

- `ownership_before/ownership_after`：ownership 搜索发生时的阶段记录。
  未进入搜索的路径不生成这两个阶段。
- `padding_before/padding_candidate/padding_after`：组件成本与已构造的物理候选，
  包括容量/分配信息未知、DMA、不单射等拒绝原因；前置资格检查排除的情况没有候选记录。
- `final_ownership_unpadded/final`：对整个函数所有参与推导的算子评价，
  前者固定最终 ownership、假设移除 UB 存储布局，后者使用真实最终布局。
  它们不等同于原始 ownership 和优化后 ownership 的比较。

每条记录尽可能包含 `estimated_latency_cycles`、`service_cycles`、
`instructions`、`requests`、`entries`、`partial`；未知指标留空，不能当成 0。
这些是每个 block 的 SIMT UB 模型指标，不能直接与整 kernel 的 us 作比例比较。
DMA、地址计算和同步成本标记为 `not_modeled`。

`ub_storage_bytes` 是所有已收集 UB 分配的保守物理字节数；
`ub_allocation_bytes` 是沿用 pass 的分配对齐取整后的总量。
各自的 `ub_extra_*` 是相对原始存储的增加量，不假设 buffer 生命周期复用。
它们是本 pass 的分配估计，不是最终设备编译器给出的实际 UB 峰值。
`buffers` 记录逻辑/物理 shape、地址映射、每个 buffer 的字节数及可识别的周期。
显式布局和 auto 使用相同的最终报告结构。

`summary.csv` 增加上述最终指标、未 padding 指标、padding 方案及 JSON 路径；
`summary.md` 增加每模式的方案/占用/成本明细表。
`candidates.csv` 汇总每个物理候选的成本和搜索状态：
`accepted_at_step` 表示当时被接受，`selected_final` 才表示最终仍采用该方案。
完整阶段和各 buffer 的映射始终保留在每个模式的 JSON 中。
