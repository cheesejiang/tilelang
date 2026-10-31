# 真实计算阶段的四模式布局实验

本目录从已有 kernel 提取八个计算阶段，比较 baseline、ownership、padding、auto。
**同一个 case 的四组输入 IR 完全相同，只改变 pass 的两个独立优化开关。**
baseline 仍执行必要的布局推导；auto 按现有规则和模型成本选择，不通过实测时延搜索。
复用 `../ub_layout_ablation/` 的诊断转换、最终 UB 分配读取、CSV/Markdown 总表。
开关默认均开启，不指定时保持原有 auto 行为。

## 文件分工

| 文件 | 作用 |
| --- | --- |
| [run_workloads.py](run_workloads.py) | 四模式编译、精度验证、计时、采集诊断、保存总表 |
| [sweep_workloads.py](sweep_workloads.py) | 固定 blocks 扫描 shape、固定 shape 扫描 blocks，生成跨配置总表 |
| [kernels.py](kernels.py) | 定义 kernel；工厂不接受 mode 参数，避免模式改变输入 IR |
| [catalog.py](catalog.py) | 场景的来源、分类、参数约束和 mode→pass 开关映射 |
| [reference.py](reference.py) | CPU 参考公式 |
| [test_runner.py](test_runner.py) | 不依赖 NPU 的参数、来源、汇总流程测试 |
| [PADDING_CANDIDATES.md](PADDING_CANDIDATES.md) | 真实 kernel 的源码核查、全部准入检查、阻塞因素和支持扩展的可行性 |

## 来源和边界

| 类别 | case | 现有代码 | 提取后的计算 |
| --- | --- | --- | --- |
| 跨行、单阶段 | gemm_scale | [epilogue.py / apply_alpha](../deepgemm/kernels/epilogue.py) | `C = 1.25 * A` |
| 跨行、单阶段 | attention_finalize | [core.py / finalize_fp32](../flash_attention/core.py) | `C[r,c] = O[r,c] / l[r]` |
| 跨行、UB 多次使用 | attention_update | [core.py / accumulate_output、kv_auto](../flash_attention/core.py) | 多个不同 KV tile：`O = alpha[kv] * O + incoming[kv]` |
| 跨行、UB 多次使用 | attention_update_finalize | [core.py / accumulate_output、finalize_fp32](../flash_attention/core.py) | KV 累加，再执行 `O/l` |
| 连续、多元素/线程 | vecadd | [example_vecadd.py / vector_add](../example_vecadd.py) | `C = A + B` |
| 连续、多元素/线程 | gemm_bias | [epilogue.py / add_bias](../deepgemm/kernels/epilogue.py) | `C[r,c] = A[r,c] + bias[c]` |
| 稀疏单列写回 | kda_column_store | [KDA / tilelang_chunk_kda_fwd_intra_token_parallel](../../kda/chunk_intra_token_parallel.py) | 只执行 `O[r,0] = reduced[r]`；归约结果预先计算 |
| 归约后单列写回 | kda_reduce_column_store | [KDA / tilelang_chunk_kda_fwd_intra_token_parallel](../../kda/chunk_intra_token_parallel.py) | kernel 内计算 `sum_d(A[r,d]*B[r,d])`，再写 `O[r,0]` |

**这是源 kernel 的阶段提取实验，不是原始端到端 kernel 的原样测试。**
vecadd 来源本来是 Ascend SIMT；GEMM/attention 来源是 SIMD，KDA 来源是 CUDA，均为阶段移植。
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

先验证单个 case，再验证全部 case、四种模式的精度：

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
| --case | all | 单个 case 或全部运行 |
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

threads 是 32 的倍数且必须整除 rows；rows 还必须是 8 的倍数，columns 为正整数。
为比较不同的行跨度，脚本不再将 columns 限制为 32 的倍数，例如允许 16、34、40。
这不保证所有形状都有优化收益；DMA/布局合法性仍由编译器检查，新列宽需要设备验证。
脚本还限制 tile 尺寸，给统计量/分配预留空间。主机输入统一为 A、incoming、Stats，
未使用的输入不在设备上搬运；CPU 参考计算保存在 `reference.py`。
增大 blocks 会增加总工作量，不是把固定数据量分给更多 block。比较模式时应保持
blocks 一致；报告的 UB 分配量是每次 block 执行所需的地址空间，不乘以 blocks。

## 批量扫描 shape 和 blocks

使用 `sweep_workloads.py`，每个配置仍调用原四模式实验，不改变 kernel 或 pass。
两个扫描维度分开控制，不做 shape×blocks 的全组合：

| 扫描 | 改变的参数 | 保持固定 |
| --- | --- | --- |
| `--sweep shapes` | `--shapes` 指定的 rows、columns | `--blocks`、`--threads`、updates、seed |
| `--sweep blocks` | `--block-counts` 指定的 blocks | `--rows`、`--columns`、`--threads`、updates、seed |
| `--sweep both` | 依次做上述两组扫描 | 各组分别控制，不互相混合 |

建议先只选 `gemm_scale` 验证流程，再扩展到全部 case。先预览配置和预计运行次数，
此命令只保存计划，不导入 NPU 运行环境，也不编译或执行 kernel：

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep both --case gemm_scale --dry-run --output layout-scan-preview
```

**第一组：固定 32 blocks 和 32 threads，扫描矩阵 shape。**
16、32、34、40 列对应不同的行跨度，可以观察模型 entry、实际布局和时延变化：

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep shapes --case gemm_scale \
  --shapes 32x16 32x32 32x34 32x40 64x32 128x32 \
  --blocks 32 --threads 32 --bench --output layout-shape-scan
```

如果希望元素总数相同、只改变行列比例，可另做一组：

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep shapes --case gemm_scale --shapes 32x128 64x64 128x32 \
  --blocks 32 --threads 32 --bench --output layout-shape-equal-area
```

**第二组：固定每 block 的 32×32 tile 和 32 threads，扫描 blocks。**

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep blocks --case gemm_scale --rows 32 --columns 32 --threads 32 \
  --block-counts 1 8 16 32 64 --bench --output layout-block-scan
```

改成 `--case all` 扫描全部 case；指定 `--mode ownership` 等可只跑一种模式，
但没有同配置 baseline 时相对加速比和 UB 增量会是未知。
不带 `--bench` 也会精度验证、采集布局及模型成本，只是不测时延。

| 扫描参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sweep` | both | shapes、blocks 或两组扫描 |
| `--shapes` | 32x16 32x32 32x34 32x40 64x32 128x32 | shape 扫描的显式列表，不是行列笛卡尔积 |
| `--blocks` | 32 | shape 扫描的固定 block 数 |
| `--rows / --columns` | 32 / 32 | block 扫描的固定 tile shape |
| `--block-counts` | 1 8 16 32 64 | block 扫描列表 |
| `--threads` | 32 | 全程固定；必须整除所有待测 rows |
| `--case / --mode` | all / all | 与单配置入口相同 |
| `--updates / --rounds / --seed` | 4 / 3 / 0 | 与单配置入口相同 |
| `--output` | layout-workload-sweeps | 跨配置报告根目录 |
| `--dry-run` | 关闭 | 只打印并保存扫描计划 |
| `--summarize-only` | 关闭 | 只读取各配置已有 JSON，重建跨配置总表 |

默认两组共 11 个配置，全部八个 case、四模式共 352 次 case/mode 运行。
所有 shape 都在启动前按预算及线程数约束检查；超出限制会报错，不会自动改变
threads 或跳过形状，以免引入额外变量。

扫描结果分别保存，彼此不覆盖：

```text
<scan-output>/
  scan_plan.json
  summary.md / summary.csv / candidates.csv     # 跨配置总表
  shapes/r32_c34_t32_b32/
    summary.md / summary.csv / candidates.csv   # 原四模式总表
    <case>/<mode>.json、.input.tir、.cpp、.tir、.allocated.tir
  blocks/r32_c32_t32_b1/
    ...
```

根目录总表启动时创建，每完成一个配置更新；子目录仍每完成一个 mode 更新。
失败配置的已有结果保留，并继续其他配置，最后返回非零退出码。
根表明确区分 scan、rows、columns、threads、blocks，不会合并不同 shape 的记录。
除原有模型/分配指标，还记录 `output_elements = blocks×rows×columns` 和
`output_melems_per_s = output_elements / latency_us`，单位 Melem/s。
这是输出元素吞吐量，不是内存带宽，也未将 KV updates 换算成操作数。

比较 block 数时，总数据量同时变化，应结合吞吐量，不能只比较绝对时延。
同配置四模式加速比才用于评价优化收益；UB 分配和模型成本是每 block 的指标。
完整矩阵的连续元素可能让 ownership 消除跨行问题；要研究固定列的稀疏访问，
请选择下述 KDA 单列写回案例，不能把完整矩阵的结论直接套过去。

补生成扫描总表时，使用与原运行相同的扫描列表和固定参数。例如：

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep blocks --case gemm_scale --rows 32 --columns 32 --threads 32 \
  --block-counts 1 8 16 32 64 --summarize-only --output layout-block-scan
```

补生成直接读取每个配置的原始 JSON，不调用设备，也不重跑子配置。
`scan_plan.json` 记录原扫描计划；当前入口要求显式传入相同扫描参数，不自动恢复计划。

## 结果

### KDA 归约后逐列写回

原 KDA kernel 每轮归约产生一个向量，再执行：

```python
T.copy(Sum_Aqk_fragment, Sum_Aqk_shared[:, j % CS])
```

该轮只能写这些行的固定列，不能像独立逐元素 kernel 那样，把待写元素任意换成
同一行的相邻列。因此新增两个不含模式专属注解的对照案例，均只改写列 0，
并检查其他列保持不变：

| case | 设备上执行 | 输入/参考结果 | 测试目的 |
| --- | --- | --- | --- |
| kda_column_store | `work[r,0] = reduced[r]` | CPU 预计算 `reduced=sum(A*B, -1)`，从 GM 搬入连续向量；其他列保留 A | 隔离单列稀疏写回，给现有 padding 搜索一个支持范围内的候选 |
| kda_reduce_column_store | `T.alloc_reducer`、点积归约、finalize，再写 `work[r,0]` | CPU 独立计算行点积；其他列保留 A | 保留归约→单列写回的数据流，观察真实约束和当前编译器边界 |

这是选取 `j % CS == 0` 的一个 epoch，不是原 KDA 完整循环。
为复用现有参数，将归约长度 DK 和输出行跨度 CS 都设为 columns；源代码的
`block_H` 默认往往比这里的 rows 小，不能将本例结果当成源默认配置的加速比。
去掉了 exp 门控、流水线、其他计算和后续列，A 同时作为已有输出矩阵和点积的
一个输入。这些简化及 CUDA→Ascend 移植都记录在 case 的 adaptation 中。

**先验证隔离写回案例**，使用四种独立开关，观察是否出现“ownership 不能消除
目标列冲突，而 padding 能降低 entry”的结果：

```bash
python examples/ascend/layout_workloads/run_workloads.py \
  --case kda_column_store --bench --output kda-column-results
```

再验证归约和写回在同一个 kernel 中的版本：

```bash
python examples/ascend/layout_workloads/run_workloads.py \
  --case kda_reduce_column_store --bench --output kda-reduce-column-results
```

当前 ownership/padding 快路径会保守排除含 reducer 的关联组件，完整版本可能
四种模式布局相同、模型指标为 `unknown`。这是当前分析能力的边界，不能解释为
padding 在硬件上无效；本次不放宽 reducer 合法性规则，不绕过同步或完整编译。
隔离版本也不保证有整体时延收益：它包含整个矩阵的 DMA 搬入/搬出，而只改一列。
是否有收益要结合精度、实际布局、entry 和整体时延一起判断。

可直接复用扫描入口，先比较固定列访问的行跨度：

```bash
python examples/ascend/layout_workloads/sweep_workloads.py \
  --sweep shapes --case kda_column_store --shapes 32x32 32x34 32x40 128x32 \
  --blocks 32 --threads 32 --bench --output kda-column-shape-scan
```

对于一个 32 活跃 lane、各写独立 f32 的请求，模型预计：32 列时原始写入 32 entry，
每 8 行加 32B 后为 8 entry；34 列原本已是 2 entry，未必值得 padding。
这是单请求几何预期，不等于整个 kernel 的成本，也不保证 pass 最终采用该周期。

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

修改 C++ 后需要先重新构建。padding 现在使用独立准入检查：固定已解出的线程和
fragment 布局，支持静态部分列 copy，并最多枚举 1024 次外层调用的实际地址。
reducer 可留在同一组件，但它直接访问的 UB 仍不能被 padding；不触及候选 UB 的
归约成本作为共同项排除。`padding_*` 诊断的 `cost_scope` 会明确标注这一范围，
不能把它当作包含归约的整核成本。详细边界见 [PADDING_CANDIDATES.md](PADDING_CANDIDATES.md)。

先确认精度、最终布局和成本是否真的改变，再比较时延。相同时延或无优化选择
都是有效结果，不把“预期收益”当成通过条件。
`--bench` 始终先验证精度，通过后立即打印 `correctness passed`，再计时。
性能状态变成 `measured` 不会覆盖 JSON/CSV 中独立的 `correctness_passed` 字段。
已有 JSON 记录该字段时，可用 `--summarize-only` 为 MD 补出精度列，无需重新跑设备。
主机测试通过不代表新协议的 C++ 开关、DSL 完整编译和设备精度已验证；
仍需在对应 Ascend 环境运行前述实验。
