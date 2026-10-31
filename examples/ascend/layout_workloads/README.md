# 真实计算阶段的四模式布局实验

本目录从已有 kernel 提取六个计算阶段，比较 baseline、ownership、padding、auto。
**同一个 case 的四组输入 IR 完全相同，只改变 pass 的两个独立优化开关。**
baseline 仍执行必要的布局推导；auto 按现有规则和模型成本选择，不通过实测时延搜索。
复用 `../ub_layout_ablation/` 的诊断转换、最终 UB 分配读取、CSV/Markdown 总表。
开关默认均开启，不指定时保持原有 auto 行为。

## 文件分工

| 文件 | 作用 |
| --- | --- |
| [run_workloads.py](run_workloads.py) | 四模式编译、精度验证、计时、采集诊断、保存总表 |
| [kernels.py](kernels.py) | 定义 kernel；工厂不接受 mode 参数，避免模式改变输入 IR |
| [catalog.py](catalog.py) | 六个场景的来源、分类、参数约束和 mode→pass 开关映射 |
| [reference.py](reference.py) | CPU 参考公式 |
| [test_runner.py](test_runner.py) | 不依赖 NPU 的参数、来源、汇总流程测试 |

## 来源和边界

| 类别 | case | 现有代码 | 提取后的计算 |
| --- | --- | --- | --- |
| 跨行、单阶段 | gemm_scale | [epilogue.py / apply_alpha](../deepgemm/kernels/epilogue.py) | `C = 1.25 * A` |
| 跨行、单阶段 | attention_finalize | [core.py / finalize_fp32](../flash_attention/core.py) | `C[r,c] = O[r,c] / l[r]` |
| 跨行、UB 多次使用 | attention_update | [core.py / accumulate_output、kv_auto](../flash_attention/core.py) | 多个不同 KV tile：`O = alpha[kv] * O + incoming[kv]` |
| 跨行、UB 多次使用 | attention_update_finalize | [core.py / accumulate_output、finalize_fp32](../flash_attention/core.py) | KV 累加，再执行 `O/l` |
| 连续、多元素/线程 | vecadd | [example_vecadd.py / vector_add](../example_vecadd.py) | `C = A + B` |
| 连续、多元素/线程 | gemm_bias | [epilogue.py / add_bias](../deepgemm/kernels/epilogue.py) | `C[r,c] = A[r,c] + bias[c]` |

**这是源 kernel 的阶段提取实验，不是原始端到端 kernel 的原样测试。**
vecadd 来源本来是 SIMT；其余来源是 SIMD，实现已改写为 SIMT。
跨行组用 `(column, row)` 逻辑循环，实际线程分配由基础推导和优化共同决定，
应查看 `loop_layouts`；不能宣称原始 SIMD kernel 存在同样的 lane 冲突。
保留源阶段的算术，但移除 Cube GEMM、NZ/BF16 打包、多版本和流水线等外围部分。
原注意力阶段的 alpha、l 作为输入给出，不在本实验重算 QK、PV、softmax 或 reduction。
每个 KV tile 使用独立的 alpha 和 incoming；因提取阶段的上游不在本实验内，
这些输入通过 GM 搬入，时延包含这些搬运，与原完整 attention 的内部交接不同。
两种多次使用案例属于同一个注意力输出路径的两个阶段组合，并非两种独立算子家族。
每份 JSON 保存来源、函数名、改写说明及源文件 SHA256。

分类只表达待验证的假设：单阶段可能收益小，多次使用可能收益更大。
跨行组现在处理完整 tile 的所有元素，而不是只取一列；即使只有一阶段，访问量也
可能足以受益。多次使用组每次搬入一个**不同的输入 tile**，同时计入这次 DMA。
不通过重复读取同一个值来制造高复用，也不保证多阶段一定比单阶段收益更大。

## 四种模式

| 模式 | tl.ascend_enable_ub_ownership | tl.ascend_enable_ub_padding |
| --- | --- | --- |
| baseline | false | false |
| ownership | true | false |
| padding | false | true |
| auto | true | true |

**必须重新编译 C++ 库**以注册开关。两个开关默认均为 true，未指定时保持原行为。
关闭的只是对应优化搜索，必要的基础布局推导及合法性检查始终执行。
同一个 case 的四组使用完全相同的无注解 kernel；跨行组始终是二维 UB，连续组
始终是一维 UB，不再通过展平 buffer、固定 float2 或显式 padding 区分模式。
kernel 工厂没有 mode 参数。每份报告记录 `pass_configs`、输入 IR 的结构哈希，
并保存 `<mode>.input.tir`，可核对四组输入。默认连续组仍是每线程多个逻辑元素。

auto 对通过现有合法性检查的组件先评估 ownership，再按选中的线程布局评估
适用的 padding 候选。二维规则跨行访问不再直接跳过 ownership 搜索。
这只是放开候选评估，不保证二维场景一定找到一维表达下的相同布局或实测收益。
合法候选中优先降低模型成本、同成本优先少占用 UB，没有收益则保留原布局。
它不穷举四种运行模式，也不保证实测全局最优。padding-only 同样通过 pass 搜索，
不再强制手工周期；不采用 padding 也是有效结果。
完整开关配置记录在 pass 诊断中。协议标识为 `identical-ir-pass-switches-v1`；
旧版展平/注解对照结果应保存在另一目录，不能与新版当作同一实验比较。
完整 `loop_layouts` 和最终 `layout_diagnostics` 均保存到 JSON。

## 运行

在仓库根目录运行，使用已构建最新 C++ 库、能导入 torch/torch_npu 的 Ascend 环境。
如果出现未知 `tl.ascend_enable_ub_*` 开关或缺少诊断属性，应先重新构建，
而不是删除开关来继续测试。建议使用新目录，避免覆盖旧版实验。

先验证单个 case，再验证全部六个 case、四种模式的精度：

```bash
python examples/ascend/layout_workloads/run_workloads.py --case vecadd --output layout-workload-results-switches
python examples/ascend/layout_workloads/run_workloads.py --output layout-workload-results-switches
```

再测量性能，默认每模式测三批，取批次时延中位数：

```bash
python examples/ascend/layout_workloads/run_workloads.py --bench --output layout-workload-results-switches
python examples/ascend/layout_workloads/run_workloads.py --case vecadd --bench --output layout-workload-results-vecadd
python examples/ascend/layout_workloads/run_workloads.py --case attention_update --updates 8 --bench --output layout-workload-results-kv8
```

不加 `--bench` 也会生成 MD、成本和分配报告，只是没有实测时延。
单跑 `--mode auto` 可以采集绝对成本及分配量，但没有同次 baseline 时不能计算
加速比或最终 UB 相对 baseline 的增量；需要四组比较时使用 `--mode all`。

可对比不同 tile 大小，但使用不同输出目录，避免覆盖前一次结果：

```bash
python examples/ascend/layout_workloads/run_workloads.py --columns 128 --bench --output layout-workload-results-c128
python examples/ascend/layout_workloads/run_workloads.py --rows 128 --threads 128 --bench --output layout-workload-results-r128
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| --case | all | 单个 case 或六个全部运行 |
| --mode | all | 单个模式或四种全部运行 |
| --rows / --columns | 32 / 32 | 每 block tile 的 shape |
| --threads | 32 | SIMT 线程数；32 的倍数，且整除 rows |
| --blocks | 32 | 启动的逻辑 block 数，每个处理一份独立 tile；不是物理核数 |
| --updates | 4 | 仅两个 attention_update 场景使用：消费多少个不同 incoming/alpha，累计更新同一个 O |
| --rounds | 3 | benchmark 批次数，各批预热 10、测量重复 30，取中位数 |
| --seed | 0 | CPU 生成数据的随机种子，各模式保持相同 |
| --bench | 关闭 | 是否测量整体 kernel 时延 |
| --output | layout-workload-results | 报告根目录 |
| --summarize-only | 关闭 | 只读取已有 JSON 重建总表，不编译、不执行、不计时 |

threads 是 32 的倍数且必须整除 rows；rows 还必须是 8 的倍数，columns 是 32 的倍数。
这些是本组规则 tile 的实验限制，不代表硬件只能处理这些形状。
脚本还限制 tile 尺寸，给统计量/分配预留空间。主机输入统一为 A、incoming、Stats，
未使用的输入不在设备上搬运；CPU 参考计算保存在 `reference.py`。
增大 blocks 会增加总工作量，不是把固定数据量分给更多 block。比较模式时应保持
blocks 一致；报告的 UB 分配量是每次 block 执行所需的地址空间，不乘以 blocks。

## 结果

每个 case/mode 单独启动进程，编译/设备故障不会污染其余模式的上下文。
失败继续收集其他模式，最终返回非零；失败原因写在对应 JSON。

```text
<output>/
  summary.md / summary.csv / candidates.csv
  cases.json
  <case>/
    <mode>.json
    <mode>.input.tir
    <mode>.cpp
    <mode>.tir
    <mode>.allocated.tir
```

总表的 iterations 列在注意力累计组表示实际 KV 更新数，其余为 1；不是对单轮
计算人为重复的次数。rows、threads、分类、来源及测量批次数据保留在 JSON。
沿用旧表的 row stride 固定为 1，它只是场景标签，不是实际 lane 地址差的测量值；
实际映射应查看 `loop_layouts` 和生成代码，不能由这一列推断。
`final IR UB allocation` 是 MergeUBAllocations 后的分配结果，不是设备实测。
时延包含 DMA、同步和 kernel 调用路径的影响，编译和 CPU 正确性验证不计时；
当前调用仍由 kernel 包装器分配输出，不能当成纯 UB 指令时延。

启动时打印 `summary.md` 的绝对路径，默认输出目录相对于**启动命令时的工作目录**，
并非相对于脚本所在位置。总表启动时创建，每完成一个 mode 就更新，不需要等四种
模式或全部 case 结束。若已有各模式的 JSON，但没有总表，可直接补生成而不跑设备：

```bash
python examples/ascend/layout_workloads/run_workloads.py --summarize-only --output /绝对路径/layout-workload-results
```

补生成使用 JSON 中记录的配置，不覆盖原始 JSON，也不重新编译或计时。
每次调用重建本次选中的 case/mode 的总表，不自动追加其他实验；正常执行会覆盖
相同 case/mode 的产物。只汇总不会给旧 JSON 补出原来未采集的指标。

## 总表指标

`summary.md` 第一张表并排显示四模式的整体时延，单位为 us；未计时时显示
正确性/执行状态，尚未运行的模式显示 `—`。`summary.csv` 还包含
`speedup_vs_baseline = baseline 时延 / 当前模式时延`，大于 1 表示更快。

第二张表显示布局和模型结果：

| 指标 | 含义和来源 |
| --- | --- |
| correctness | 独立的精度状态：`passed` 表示全部输出已与 CPU reference 比较通过；未采集/未完成时为 `unknown`，不由 measured 状态推断 |
| padding scheme | 最终布局识别出的周期和 padding 字节数；`none_or_unrecognized` 表示未识别到周期方案，应结合 JSON 核对 |
| extra UB (B) | 布局物理大小相对原始存储增加的字节数，由布局 pass 计算 |
| estimated extra allocation (B) | pass 将各 buffer 按 32B 取整后累加，不计别名复用，计算出的分配增量 |
| final IR UB allocation (B) | 从 MergeUBAllocations 后读取的普通 UB 分配量，不再次取整；同函数分配累加，多函数汇总取最大值 |
| final IR extra vs baseline (B) | 同配置当前模式最终 IR 分配量减去 baseline 的最终 IR 分配量 |
| UB latency (cycles) | 当前模型估计的 SIMT UB 访问完成成本，含已建模的循环次数 |
| service (cycles) | entry 发布流水线的模型服务周期数，不是整个 kernel 时延 |
| instructions | 模型估计的 LDS/STS 指令实例数，不是最终 ISA 指令统计 |
| requests | 每条访存指令拆分出的请求总数，每请求每 lane 至多处理 4B；b64 分成两请求 |
| entries | 请求按 key/group/line 和容量规则打包后，需要发布的 entry 总数 |
| partial | 不完整 4B 字写入的 entry 数，用于标识可能需要读改写的访问 |

前两种存储指标是前期计算值；final IR 指标已经包含该分配器的对齐及别名复用，
但仍不是 Bisheng 资源报告或设备运行时测量。缺快照或动态分配大小显示 `unknown`。
缺 baseline、基线配置不匹配或实验协议不同时，不计算相对 baseline 的增量/加速比。
模型成本不完整包含 DMA、同步、地址计算、寄存器压力等开销，不能直接把 cycles
等同于实测 us，也不能把 entry 降低百分比当作整个 kernel 的加速百分比。

`candidates.csv` 是已构造 padding 候选的明细；前置资格检查直接排除的场景不一定
有候选行。`accepted_at_step` 表示搜索某一步接受，`selected_final` 才表示最终采用。
`unpadded_*` 成本字段固定最终 ownership、假设移除 UB 存储布局，用于隔离 padding
影响，不等同于 baseline 或原始 ownership 的成本。

每模式 JSON 还保留 `experiment_protocol`、`pass_configs`、`input_ir_structural_hash`、
`loop_layouts`、完整 `layout_diagnostics`、最终分配明细 `final_ir_ub`、
`latency_samples_us` 和源代码信息。解释优化效果时先检查四组输入和开关，再检查
实际布局、模型成本是否改变，最后比较实测时延。

## 检查脚本

主机侧测试，不需要 NPU；NumPy 未安装时会跳过 CPU 参考公式检查：

```bash
python -m unittest discover -s examples/ascend/layout_workloads -p 'test_*.py' -v
```

在有 TVM、pytest 和最新编译库的环境中，检查输入 IR 一致性和 pass 开关：

```bash
python -m pytest testing/ascend/transform/test_ascend_layout_workload_modes.py testing/ascend/transform/test_ascend_simtvf_ub_padding.py -q
```

先确认精度、最终布局和成本是否真的改变，再比较时延。相同时延或无优化选择
都是有效结果，不把“预期收益”当成通过条件。
`--bench` 始终先验证精度，通过后立即打印 `correctness passed`，再计时。
性能状态变成 `measured` 不会覆盖 JSON/CSV 中独立的 `correctness_passed` 字段。
已有 JSON 记录该字段时，可用 `--summarize-only` 为 MD 补出精度列，无需重新跑设备。
主机测试通过不代表新协议的 C++ 开关、DSL 完整编译和设备精度已验证；
仍需在对应 Ascend 环境运行前述实验。
