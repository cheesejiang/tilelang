# Ascend950 多型号适配、优化与测试：会话记忆

更新时间：2026-11-16。本文记录需求、讨论结论、已有代码及证据边界，供后续工作接续；不代表所有结论都已在所有型号上实测。

## 1. 用户需求及工作约束

- 目标：让当前 TileLang Ascend 后端在指定的 31 个 Ascend950 具体型号上可用，借鉴 Triton 的通用架构/规格建模思路，并设计正确性转测与性能对照方案。
- 当前只有 Ascend950PR_9579 可执行上板实验；用户在另一台机器运行代码并反馈结果。不要自行猜测 SSH 地址或连接 950DT；应提供代码和运行命令。
- 用户希望先讲清方案，明确要求修改后再改代码；不要将讨论中的优化建议直接当作已实现功能。
- 重点区分：功能可用、数值正确、性能可接受、性能无退化。这些不是同一个结论。
- 当前不考虑 GM 自动拆分；优化讨论集中于逻辑 blocks 数与 L2 数据复用/遍历分组。

## 2. 目标设备与硬件依据

讨论的 31 个具体型号：

```text
Ascend950DT_950x  Ascend950DT_950y  Ascend950DT_9571
Ascend950DT_9572  Ascend950DT_9573  Ascend950DT_9574
Ascend950DT_9575  Ascend950DT_9576  Ascend950DT_9577
Ascend950DT_9578  Ascend950DT_9581  Ascend950DT_9582
Ascend950DT_9582x Ascend950DT_9583  Ascend950DT_9584
Ascend950DT_9585  Ascend950DT_9586  Ascend950DT_9587
Ascend950DT_9588  Ascend950DT_9591  Ascend950DT_9592
Ascend950DT_9595  Ascend950DT_9596  Ascend950DT_95A1
Ascend950DT_95A2  Ascend950PR_950z  Ascend950PR_9579
Ascend950PR_957b  Ascend950PR_957bx Ascend950PR_957c
Ascend950PR_957d
```

- 先前讨论结合 `/home/Ascend/cann` 型号 ini 及本地 NPUIR 源码分析：这些设备具有共同架构与基础核内内存规格，但核数、L2、GM 容量、带宽等不应视为全部相同。后续若要正式出具完整规格表，应再次定位原始 ini/源码，保留逐项比较证据。
- 共同字段包括：Short_SoC_version=Ascend950，AIC_version=AIC-C-310，CCEC_AIC_version=dav-c310-cube，CCEC_AIV_version=dav-c310-vec，NpuArch=3510，Chip_type=15；L0A=65536、L0B=65536、L0C=262144、L1=524288、UB=253952、BT=4096 字节；vector_reg_width=256、predicate_reg_width=32。wide_reg_width 空值不能当作实际数值规格。
- 已讨论的 UB 字段：ubblock_size=32、ubbank_size=4096、ubbank_num=16、ubsubbank_num=64、ubburst_in_one_block=32、ubbank_group_num=8。不要把这些字段简单相乘推导 UB 总容量，也不要无文档依据解释 burst 的具体微架构意义。
- NPUIR 先前源码分析认为这些具体型号继承 Ascend950_BaseSpec，仅覆盖三个核数字段。它是参考证据，不是当前 TileLang 后端实际使用该模型的证明；更不能由该建模推出 L2/GM/带宽相同。
- Triton 参考 PR：NPUIR https://gitcode.com/Ascend/AscendNPU-IR/pull/4285；TA https://github.com/triton-lang/triton-ascend/pull/2776 。不要仅凭 isAscend950 的具体枚举名单判定新通用入口的全部行为。

## 3. 当前后端与 blocks 的结论

- 当前选定 `target=ascend`、`execution_backend=tvm_ffi`，走 Ascend C++/Bisheng 链路，默认架构 dav-3510；不是因为 NPUIR 已列举 SKU 就自动获得支持。
- 通用流程：DSL/PrimFunc → TileLang lowering/codegen → Ascend 源码 → Bisheng 编译 → 运行时加载及提交 → NPU 执行。JIT 包装与显式先 compile 再调用主要区别在编译组织方式；最后进入相同执行链路。不要混淆 `.so` 文件与用户所说的分阶段 DSL 构造方式；具体调用以仓库实例为准。
- blocks/grid 表示逻辑工作组织，不等于必须拥有同样数量的物理核。一个 block 的索引参与任务与地址映射。
- 用户报告：9579 上 JIT 与分阶段测试全部通过，包括 blocks 超过实际核数的用例。可以说明被测 kernel 在这些配置下正确执行、下层能够完成逻辑工作分发；尚未用 profiler/运行时源码精确确定排队由哪一层负责。
- 不能因此声称任意 DSL、任意 blocks 都合法。仅对采用合法任务分配方式的 kernel 改 blocks 才能保持覆盖完整：例如 task_id=bx, bx+B, bx+2B...，并处理尾轮。
- 改 blocks 若与原任务划分不匹配，可能遗漏、重复或越界；硬件自动分发不会修正 DSL 的索引错误。
- 架构兼容前提下，目前倾向无需为这 31 个 SKU 增加具体型号白名单。若需要 A5 识别，可集中在一个函数中按 Ascend950 前缀判断并保留例外入口；这只是建议，不能写成已经改入框架。
- 无需仅为运行正确性强制把物理核数传给每个 kernel。性能配置是否需要感知核数应单独分析。

## 4. 优化方向与边界

### 逻辑 blocks 数

- 同一工作量下扫不同 blocks，观察并行度、尾轮、调度开销、搬运与计算重叠的影响。
- 重点候选 C-1、C、C+1、2C，另保留原始默认值；C 对纯向量用 AIV 数，对 Cube GEMM 用 Cube 数，混合核需要结合实际执行模式，不能套用一个泛称 AI core 数。
- 并非 blocks=C 一定最优。任务数量、tile 大小、流水深度和数据复用都会影响最优值。
- 不能全局无条件重写 DSL 中的 blocks。要先证明工作映射可重参数化，或由算子生成器显式暴露参数。
- 候选配置 + 成本模型/自动调优，是框架选择具体算子配置的可能方案，不代表当前已具备通用自动优化 Pass，也无法免除合法性分析。

### L2 分组与遍历顺序

- 对可分析的数据复用关系，改变逻辑 tile 的遍历顺序或分组，让相邻任务复用 A/B 等数据；不改变数学结果。
- 不等同于修改 UB/L1/L0 容量，不等同于 GM 不够时自动拆分大任务。
- 当前阶段先测工作集与复用敏感性，不承诺实现通用重排 Pass，也不承诺所有 kernel 均可自动识别和优化。

### GM、带宽与跨 SKU

- GM 容量主要影响数据能否分配；容量相同不代表带宽相同。L2 容量、DRAM 带宽、频率等影响性能。
- 在 9579 减少 blocks 不能模拟另一 SKU；更大工作集也不等于修改实际 L2 容量。
- 没有其他 SKU 的结果，无法证明这些型号无性能退化。9579 的实验用于建立参考及识别敏感性。

## 5. 硬件敏感性实验（已提供代码）

目录：`examples/ascend/hardware_sensitivity/`。

- `benchmark.py`：blocks、reuse、stream 三组；采集环境、CSV、生成源码，可离线画图。
- blocks：Copy、VecAdd、简单持久化 GEMM；固定工作量扫描逻辑 blocks。
- reuse：多个工作集容量，重复遍历，测缓存复用相关敏感性。
- stream：单次遍历工作集；没有主动清缓存，不能称为严格冷缓存 DRAM 带宽。
- 逻辑带宽=逻辑读写字节/耗时，不是 profiler 测到的 DRAM 流量，可能超过物理 DRAM 带宽。
- `real_operators.py`：复用原始 VecAdd/GEMM 算子，补 C-1/C/C+1/2C 与原默认配置。
- 原 `example_vecadd.vector_add` 增加可选 num_blocks（默认64）并处理不完整 block 轮次；仍要求元素数满足原 tile 粒度。原 `example_gemm.gemm` 增加可选 num_blocks（默认32）。这些是更早的实验改动，不是本次 FP8 对照替换改动。
- 编译、参考计算、验证在敏感性实验计时之外；预热后用 NPU events，多次重复采样记录 median/min/max。多轮平均值不是单次调用延迟分位数。
- 用户已运行并反馈过敏感性及真实算子实验。准确数值以其原始 CSV 为准，不能在新会话中杜撰结果。

## 6. 正确性转测范围

以 `examples/ascend` 原本的算子为基线；排除 ub_layout_ablation、layout_workloads。独立 register-store 示例不再单列，VecAdd SIMD 已覆盖相应 GM→UB→寄存器→UB→GM 基础路径，但不表示所有寄存器指令均被穷尽覆盖。

| ID | 功能 | 入口与重点 |
|---|---|---|
| C01 | VecAdd | test_vecadd.py；SIMT/SIMD、默认64 blocks、非整除任务轮次 |
| C02 | RMSNorm | test_rmsnorm.py；d=4096/5120/7168、缓冲区复用 |
| C03 | FP8 group quantization | test_per_token_cast_to_fp8.py；SIMT/SIMD、group128、量化值与FP32 scale、全零行 |
| C04 | GEMM | test_gemm.py；FP8/BF16/FP32/HF32、Cube/混合、输出与累加、持久化尾轮 |
| C05 | Split-K | test_gemm_splitk.py；atomic/deterministic、完整输出、确定性路径重复一致 |
| C06 | DeepGEMM | deepgemm/test_deepgemm.py；dense/BMM/M-K grouped、scale与布局、epilogue等；未测公开API组合记缺口 |
| C07 | MHA forward | flash_attention/test_mha.py |
| C08 | GQA forward | flash_attention/test_gqa.py |
| C09 | GQA manual schedule | flash_attention/test_gqa_manual_schedule.py；包装原 check_correctness，手动调度不能由自动版代替 |
| C10 | GQA backward | flash_attention/test_gqa_bwd.py；Delta、dQ/dK/dV、短流水、缓冲复用、非二次幂KV tile数 |

- core.py/core_bwd.py、DeepGEMM kernel/helper 通过入口间接覆盖，不声称所有 API/形状均已穷尽。
- 各入口独立进程与外部超时；逐项记录通过、失败、skip、不支持及覆盖缺口，skip 不能算通过。
- 仅跑少量核心组可以作为冒烟，不足以证明全部 SKU 完整适配。

## 7. 性能基线原则

- 无同卡历史版本结果时，不能声称“无退化”。Torch-NPU 对照衡量相对原生实现的性能竞争力，不是版本回归证明。
- 真正回归需固定同一设备、软件栈、输入、精度、配置、计时方式，比较改前/改后，必要时多会话重复并分析波动。
- 记录输入形状、dtype、输出范围、数学语义、精度模式、配置与计时边界；同类至少选代表算子，但不能把不等价对照强配成 fused baseline。
- 对齐输出（如 RMSNorm Y/RSTD、attention backward 所有梯度）、精度、输入布局；预处理包含/排除需双方一致且明确。
- 用户要求剔除仅相对组合基线显著占优的性能项：Split-K、FP8 group quantization、rmsnorm_composite。它们仍保留正确性覆盖。RMSNorm 性能只保留 fused 对照。

## 8. 当前 eager 性能脚本

目录：`examples/ascend/eager_baseline/`。

文件：`compare.py`、`README.md`、`BASELINE_PLAN.md`、`baseline_inventory.json`、`test_compare_driver.py`。实际文件存在与内容以当前 checkout 为准。

当前14个 case：

```text
vecadd_simt vecadd_simd rmsnorm_fused
gemm_bf16 gemm_fp32 gemm_mixed_fp32
deepgemm_bf16 deepgemm_bmm deepgemm_grouped
deepgemm_fp8_quant_matmul
mha_bf16 gqa gqa_manual gqa_backward
```

- 对照使用 PyTorch eager：直接执行 torch/torch_npu 算子，没有 torch.compile 或图编译。
- eager 是执行模式，不表示一定由多个小算子组合。torch_npu 原生接口的底层是否走 ops-nn/Ascend C，需按具体安装版本追踪，不能一概而论。
- 独立子进程、外部超时、单例日志/JSON、results.csv、environment.json；保存生成源码。
- 运行目录避免复用旧成功 JSON；崩溃/超时不能误报上次通过。
- 预热与首次初始化后交替计时，前后检查正确性。接口不存在记 baseline_unavailable，不给性能通过结论，不静默替换成其他算法。
- speedup=eager耗时/TileLang耗时；>1 表示 TileLang 更快。

运行（TileLang仓库根目录、已构建的Ascend环境）：

```bash
python examples/ascend/eager_baseline/compare.py --profile smoke --output eager-all-smoke
python examples/ascend/eager_baseline/compare.py --cases deepgemm_fp8_quant_matmul --profile smoke --output eager-fp8-smoke
```

用户旧版19项 smoke报告18 passed、1 failed（旧 deepgemm_fp8_composite）。其中若干项已按用户要求删除，不能视为当前14项全部通过，也不能将旧结果挪给新的FP8用例。

## 9. FP8 对照替换：前后流程与 scale

### 旧用例 deepgemm_fp8_composite

1. 生成随机数乘0.125，再转换 FP8 E4M3FN：A[M,K]、B[N,K]。
2. scale 全1；E8M0字节127编码1，两个字节打包为int16的0x7F7F。这个整数是编码容器，不是缩放倍数32639。
3. TileLang调用 fp8_gemm_nt，输出FP32。
4. 对照执行 torch.mm(a.float(),b.float().T)，包含显式转换，实际是FP32组合路径，不是原生FP8对照。
5. 旧包装scale使用[K/64,M]、[K/64,N]，不符合公共API逻辑形状。用户报告失败但未提供该case traceback，不能断言这是唯一失败原因。

### 新用例 deepgemm_fp8_quant_matmul

- 仍使用FP8 E4M3FN输入；smoke M=N=K=256，full为1024（如脚本后续更改以实码为准）。
- scale按行与K32分组变化，取0.5/1/2，对应E8M0编码126/127/128，A/B排列错位。
- 数学语义：C[m,n]=Σk(A[m,k]×sA[m,floor(k/32)])×(B[n,k]×sB[n,floor(k/32)])。
- 输入是合成的已量化数值及合法scale；不是由高精度源数据统计出的真实量化scale，不测端到端量化精度/开销。
- 非单位scale用于发现布局、索引、编码/漏乘错误；并非原生接口强制要求，也不是性能优化。

双方接口布局：

| 对象 | TileLang公共API | Torch-NPU接口 |
|---|---|---|
| A数据 | [M,K] FP8 | [M,K] FP8 |
| B数据 | [N,K] FP8 | [K,N] FP8（b.T视图） |
| A scale | [M,K/64] int16，stride(1,M) | [M,K/64,2] uint8编码 |
| B scale | [N,K/64] int16，stride(1,N) | [K/64,N,2] uint8编码 |

- int16打包两个相邻K32 scale；`.view(int16)`是字节重新解释，不是数值转换。
- TileLang公共NT接口保持B输入形状，内部_prepare_scales再转成kernel的[K/64,M]、[K/64,N]。不要把kernel布局直接传给公共接口。
- 上一版曾错误地为公共API返回sb.T；2026-11-16已修正为sb，并修正文档与CPU布局测试。不能继续传播B scale=[K/64,N]作为公共接口要求。

调用：

```python
api.fp8_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, 32))

torch_npu.npu_quant_matmul(
    a, b.T, native_sb,
    pertoken_scale=native_sa,
    pertoken_scale_dtype=e8m0,
    scale_dtype=e8m0,
    group_sizes=[1, 1, 32],
    output_dtype=torch.float32,
)
```

- 这是eager下直接调用原生NPU FP8接口；未核实实际安装版本底层是否具体为Ascend C，因此不要直接称“已确认Ascend C实现”。
- 官方参考：https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md 。版本与格式限制必须上板检查；没有静默FP32 fallback。
- 独立CPU FP32参考：将FP8转FP32，按K32展开scale并逐元素相乘，再矩阵乘；分别检查两侧结果，rtol=atol=1e-3。CPU参考不计时。
- 输入生成、FP8初次转换、scale生成/打包/布局、B转置视图准备、CPU参考、首次编译初始化都在正式计时之外；两侧输出分配保留在调用函数中。
- NPU event时间不等同于完整Python主机开销；需要端到端开销时结合wall模式。
- 本次仅修改性能包装、测试、文档；未改原始DeepGEMM kernel/API。历史VecAdd/GEMM参数化改动另见第5节。
- 本地 `/.venv` 对应工作区 `.venv/bin/python` 跑 driver unittest 4项通过，覆盖scale字节/布局、配置、旧结果隔离、超时处理；没有本地NPU执行证据。新FP8用例待用户9579验证。

## 10. 下一步

1. 用户在9579运行新的FP8单case，提供case日志、结果CSV及environment.json；失败先按接口参数/版本/布局定位，不以旧结果代替。
2. 对当前14项跑smoke再跑正式形状，检查对照数学语义和精度设置；区分正确性通过与性能结论。
3. 保留9579可复现基线，记录原始配置与blocks候选表现；不要以自动调优后的结果冒充原默认配置性能。
4. 有其他SKU后按同一正确性集合与性能矩阵实测，才能验证跨型号适配与性能。
5. 如后续需要全局自动配置，先确认可合法参数化的任务映射与后端接口，再实现候选生成/选择，避免无条件改blocks或夸大通用Pass能力。

## 11. FP8 上板新增错误与修正

用户报告原生接口报错 `Input x2 tensor and scale tensor's transpose are not same`。包装中x2=b.T保留转置视图，但native_sb在transpose后调用contiguous，导致转置布局不一致。已删除B scale的contiguous，让它与x2同为转置视图，并增加CPU stride检查（B scale stride=(2,K/32,1)）。这是测试包装修正，未改原始算子；修正后仍待9579复测。

## 12. full形状更新

按用户要求，full对齐原始性能入口：VecAdd N=2^30；RMSNorm batch4096,d4096；GEMM8192³；MHA S4096,D128；GQA/manual/backward S1=8192,G32,S2=8192,D128。DeepGEMM dense按原性能回归清单：BF164096³，FP8 M1024,N4096,K7168。BMM/grouped没有对应固定原性能入口，保留groups8、每组1024³作为代表形状并明确标注。smoke不变；仅对齐形状，原对照dtype/语义不变。旧full结果不能与新full混作同一工作负载。用户已确认新FP8用例在9579复测通过（该确认对应此次扩大full形状之前）。

## 13. 默认三轮独立转测

用户要求自动三轮。compare.py新增--rounds默认3，每轮新suite进程，每用例新worker；每轮独立目录保存环境和日志，根results.csv保留所有轮次，summary.csv对各轮median耗时取median再计算eager/TL比值，保留逐轮比值min/max。任一轮失败/缺失/对照不可用不生成有效汇总加速比；--rounds1可单轮。保留原预热10、5组×20次event协议。已检查本地TileOPs Ascend方法：预热3、采样15、每次1GiB清缓存、profiler设备busy区间并集、均值；其计时口径不同，当前未声称复现该协议。新增三轮成功和失败汇总主机测试，5项测试通过；未在本机NPU运行。跨型号保持率仍未自动计算。

## 14. 改用TileOPs式profiler计时

用户明确要求改代码：默认timing=profiler，warmup3、samples15、repeats1、三轮独立进程。每次预热/正式调用前1GiB缓存驱逐并同步，清缓存位于profiler窗口外；窗口内一次完整调用，统计全部kernel执行区间并集，避免重叠重复计数。单轮以15样本均值为latency，加速比=eager_latency/TL_latency；三轮对各轮均值取均值。保留median/min/max/stdev和原始trace、解析记录。新增profile_parser.py/profiler_timer.py适配本地TileOPs方法；parser依赖CANN9.2 FFTS格式，缺记录或清缓存混入会失败，不回退event。未导入TileOPs预约与锁频机制；旧event/wall仍可显式选择，不能混用结果。8项主机测试通过；新profiler路径待上板；需重建9579性能基线。
