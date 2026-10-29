# UB ownership / padding 四状态对比

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
| `row1` | 32×32 | 每行第 0 列，32 个线程 | 每行插入 8B |
| `row16` | 512×32 | 每 16 行的第 0 列，32 个线程 | 每 16 行插入 8B |

跨行场景 baseline 的每 lane 一个元素，初始化仍覆盖完整 UB。
周期布局为 `F(i) = i + floor(i / K) * p`，K 是周期元素数，p 是 padding 元素数。
`row1` 使用 K=32、p=2，`row16` 使用 K=512、p=2。
不在最后一个元素之后额外分配无用尾部空洞。

## 为什么分分析和测量两个 kernel

目前自动 padding 会排除含 GM↔UB DMA 的组件。直接把 DMA 输入输出加到
自动布局例子里，会导致根本不触发 padding。

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
