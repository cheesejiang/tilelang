# 真实计算阶段的四模式布局实验

本目录替代“对同一份 UB 数据重复做虚拟计算”的收益放大方式，选择现有 kernel
中的计算阶段。每类两个例子，均有 baseline、ownership、padding、auto。
复用 `../ub_layout_ablation/` 的诊断转换、最终 UB 分配读取、CSV/Markdown 总表。
不修改 pass 的优化规则。

## 来源和边界

| 类别 | case | 现有代码 | 提取后的计算 |
| --- | --- | --- | --- |
| 跨行、单阶段 | gemm_scale | [epilogue.py / apply_alpha](../deepgemm/kernels/epilogue.py) | `C = 1.25 * A` |
| 跨行、单阶段 | attention_finalize | [core.py / finalize_fp32](../flash_attention/core.py) | `C[r,c] = O[r,c] / l[r]` |
| 跨行、UB 多次使用 | attention_update | [core.py / accumulate_output、kv_auto](../flash_attention/core.py) | 多个不同 KV tile：`O = alpha * O + incoming[kv]` |
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

auto 保持规则驱动：二维规则跨行访问保留基础 ownership，搜索物理 padding；
其他支持场景评估 ownership，然后评估适用的 padding 候选。
合法候选中优先降低模型成本、同成本优先少占用 UB，没有收益则保留原布局。
它不穷举四种运行模式，也不保证实测全局最优。padding-only 同样通过 pass 搜索，
不再强制手工周期；不采用 padding 也是有效结果。
完整开关配置记录在 pass 诊断中。协议标识为 `identical-ir-pass-switches-v1`；
旧版展平/注解对照结果应保存在另一目录，不能与新版当作同一实验比较。
完整 `loop_layouts` 和最终 `layout_diagnostics` 均保存到 JSON。

## 运行

先验证全部六个 case、四种模式的精度：

```bash
python examples/ascend/layout_workloads/run_workloads.py
```

再测量性能，默认每模式测三批，取批次时延中位数：

```bash
python examples/ascend/layout_workloads/run_workloads.py --bench
python examples/ascend/layout_workloads/run_workloads.py --case vecadd --bench
python examples/ascend/layout_workloads/run_workloads.py --case attention_update --updates 8 --bench
```

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
| --blocks | 32 | 每次 launch 的 block 数 |
| --updates | 4 | 注意力累计多少个不同 KV 输出 tile；其他 case 不使用 |
| --rounds | 3 | benchmark 批次数，各批预热 10、测量重复 30，取中位数 |
| --seed | 0 | CPU 生成数据的随机种子，各模式保持相同 |
| --bench | 关闭 | 是否测量整体 kernel 时延 |
| --output | layout-workload-results | 报告根目录 |

columns 必须是 32 的倍数，保证这组显式 32B DMA 布局的规则性。
限制 tile 尺寸给统计量/分配预留空间。主机输入统一为 A、incoming、Stats，
未使用的输入不在设备上搬运；CPU 参考计算保存在 `reference.py`。

## 结果

每个 case/mode 单独启动进程，编译/设备故障不会污染其余模式的上下文。
失败继续收集其他模式，最终返回非零；失败原因写在对应 JSON。

```text
layout-workload-results/
  summary.md / summary.csv / candidates.csv
  cases.json
  <case>/
    <mode>.json
    <mode>.cpp
    <mode>.tir
    <mode>.allocated.tir
```

总表的 iterations 列在注意力累计组表示实际 KV 更新数，其余为 1；不是对单轮
计算人为重复的次数。rows、threads、分类、来源及测量批次数据保留在 JSON。
沿用旧表的 row stride 为 1；连续组的实际相邻 lane 元素差应结合 float2/所选
ownership 理解，不能由这一列推断。
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

先确认精度、最终布局和成本是否真的改变，再比较时延。相同时延或无优化选择
都是有效结果，不把“预期收益”当成通过条件。
本地缺少 TVM/CANN/NPU，未验证新 DSL kernel 的编译、设备精度或性能。
