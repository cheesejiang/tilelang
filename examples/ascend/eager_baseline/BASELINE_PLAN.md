# Ascend 算子转测基线

## 范围与状态

目标是当前 Ascend C++ → Bisheng 后端，不选择 PTO，不修改编译器框架，
不涉及 GM 自动分块。排除 `ub_layout_ablation/` 和 `layout_workloads/`；其余原有
算子进入正确性范围。性能参照统一使用 **Torch-NPU eager 基线**：Torch 张量
位于 NPU，通过 PyTorch/torch_npu 逐算子调用，不使用 torch.compile/TorchAir
图编译。此文件是转测清单和接口选择；已有初版运行器见
[`eager_baseline/README.md`](eager_baseline/README.md)，接口仍待上板验证。

Torch-NPU eager 是框架调用层级，不保证底层实现都来自 ops-nn。若需要 ops-nn
或其他底层库归属，另查具体版本的分发代码和 profiler，不根据 API 名称推断。

`baseline_inventory.json` 逐文件列出当前目录原有 Python 代码的角色、正确性
入口及性能归属。`ub_layout_ablation/`、`layout_workloads/` 不属于本轮转测范围；
`hardware_sensitivity/` 是本次新增实验，也不纳入原有算子分母。
`example_simd_register_store.py` 的数据通路由 VecAdd SIMD 覆盖，不单列转测项。
辅助模块通过其完整算子入口验证，不把每个 Python 函数都当作独立算子。
新增文件、公开 API 或新的 dtype/布局分支应补充清单。

## 1. 正确性：原有算子全覆盖

| ID | 算子/功能 | 覆盖范围 | 已有入口 |
|---|---|---|---|
| C01 | VecAdd | SIMT、SIMD；默认 block；新增非整除任务轮次 | `test_vecadd.py` |
| C02 | RMSNorm | Y 和 RSTD；d=4096/5120/7168；复用缓冲区 | `test_rmsnorm.py` |
| C03 | FP8 group quantization | SIMT、SIMD；group=128；量化值和 FP32 scale；全零行 | `test_per_token_cast_to_fp8.py` |
| C04 | GEMM | FP8、BF16、FP32、HF32；Cube/混合；BF16/FP32 输出；累加；持久化尾轮 | `test_gemm.py` |
| C05 | Split-K GEMM | 原子、deterministic 两条路径；检查完整输出；确定性路径重复一致 | `test_gemm_splitk.py` |
| C06 | DeepGEMM | BF16/FP8 dense、BMM、M/K grouped、布局、scale 转换、alpha/bias/累加、量化 epilogue、随机舍入 | `deepgemm/test_deepgemm.py`；公开 API 的未测组合另记覆盖缺口 |
| C07 | MHA forward | BF16、输出精度 | `flash_attention/test_mha.py` |
| C08 | GQA forward | shared KV、多 query heads、输出精度 | `flash_attention/test_gqa.py` |
| C09 | GQA manual schedule | 同算子的手动调度版本，不得被自动版本替代 | `flash_attention/test_gqa_manual_schedule.py` |
| C10 | GQA backward | Delta、dQ/dK/dV；短流水、复用和非二次幂 KV tile 数 | `flash_attention/test_gqa_bwd.py` |

`core.py`、`core_bwd.py`、DeepGEMM 的 kernels 和 API helper 通过上述入口覆盖。
已有 pytest 是起点，不声称所有导出 API、所有形状组合已被现有测试穷尽覆盖。
转测前逐项记录“已有覆盖/需要补用例/不支持”，不将 skip 计为通过。

先运行已有正确性测试（在已构建的仓库环境中）：

```bash
TILELANG_DEFAULT_TARGET=ascend TILELANG_EXECUTION_BACKEND=tvm_ffi \
python -m pytest examples/ascend/test_vecadd.py \
  examples/ascend/test_gemm.py examples/ascend/test_gemm_splitk.py \
  examples/ascend/test_rmsnorm.py examples/ascend/test_per_token_cast_to_fp8.py \
  examples/ascend/deepgemm/test_deepgemm.py \
  examples/ascend/flash_attention/test_mha.py \
  examples/ascend/flash_attention/test_gqa.py \
  examples/ascend/flash_attention/test_gqa_manual_schedule.py \
  examples/ascend/flash_attention/test_gqa_bwd.py -v
```

C09 手动调度版本已纳入上述 pytest 命令，无需单独补跑示例。正式批量转测时，
建议各入口使用独立进程及外部超时，避免设备错误、编译器 abort 或死锁使其余
结果丢失。

高精度数学参考与“Torch-NPU eager 性能参照”分开：精度沿用现有 reference/阈值；
另检查输出 finite、辅助输出、任务覆盖。不可因 eager 输出数值相近而统一放宽阈值。

## 2. 性能：Torch-NPU eager 对照清单

### 2.1 对照类型

| comparison_type | 含义 | 报告名称 |
|---|---|---|
| `eager_standard` | NPU 上的 torch.add/mm/bmm 等标准接口 | Torch-NPU eager 标准算子 |
| `eager_fused` | eager 调用 torch_npu 的融合接口 | Torch-NPU eager 融合算子 |
| `eager_composite` | 多个 NPU 操作共同完成相同功能 | Torch-NPU eager 多算子组合 |
| `none` | 安装版本没有合法且语义等价的 eager 对照 | 无等价性能对照，仅记录 TileLang 基线 |

三种有效对照分栏报告，不把多算子组合标成单一融合算子。所有参照输入和输出
必须在 NPU 上；使用 profiler 记录实际设备算子，不允许 CPU fallback。
直接调用融合 API 仍是 eager 调用，不要求将融合算子展开成基础操作。

### 2.2 代表算子、接口和无对应接口时的处理

下表接口是选定的对照入口，**仍须在目标 Torch-NPU/CANN 版本核验可用性、
支持的 dtype/shape、输出和数值语义**，不能仅凭文档存在就标记“已验证”。

| ID/类别 | TileLang 代表 | 首选 Torch-NPU eager 对照及类型 | 第一版形状 | 没有等价首选接口时怎么处理 |
|---|---|---|---|---|
| P01 逐元素/访存 | VecAdd，SIMT/SIMD 分开记录 | `torch.add(a, b, out=out)`，`eager_standard` | N=2^20、2^24、2^26；FP32 | NPU add 不支持该 dtype/shape 时记录不支持，不改 dtype 或转 CPU；保留 TL 耗时 |
| P02 规约/归一化 | RMSNorm，包含 Y/RSTD | `torch_npu.npu_rms_norm(x, weight, epsilon)`，`eager_fused` | batch=128/4096，d=4096；另加 d=7168；FP32 | 使用 FP32 NPU `square→mean→add(eps)→rsqrt→mul(x)→mul(weight)`，同时返回 RSTD，标 `eager_composite`；组合也不支持则 `none` |
| P03 dense Cube/混合 | GEMM：BF16 Cube/BF16 输出、FP32 Cube/FP32 输出、FP32 mixed/FP32 输出 | `torch.mm(a, b.T)`，`eager_standard`；可用 out 版本时预分配 | 2048³、4096³；M=256,N=4096,K=7168 | 仅比较双方支持且输入输出/累加精度一致的配置。BF16 输入/FP32 输出 mixed 如无保持 FP32 累加输出的接口，只记录 TL 基线；不以 BF16 mm 后 cast 替代 |
| P04 多分区规约 | Split-K：原子、确定性分别记录 | `torch.mm(a, b.T)`，`eager_standard`，不要求 eager 也采用 Split-K | M=N=512,K=8192；对齐现有 BF16 输入/FP32 输出 | 先核验是否有保持该输出精度的 NPU matmul 接口；无则将同一 BF16 输入在计时区内转 FP32，再 FP32 mm，标 `eager_composite`，不得称 BF16 原生矩阵乘性能。若无法满足数值要求则 `none` |
| P05 grouped/batched GEMM | DeepGEMM BF16 grouped、BF16 BMM | grouped：`torch_npu.npu_grouped_matmul`，`eager_fused`；BMM：`torch.bmm`，`eager_standard` | groups=8，每组 M=128/512,N=K=1024；BMM B=8 | grouped 接口缺失或不支持该布局时，按真实 group 边界切片，逐组 `torch.mm` 并写回，标 `eager_composite`；BMM 如需循环 mm 也标组合。保持输出精度；必要 cast 计时，禁止默默降低为 BF16 输出 |
| P06 低精度矩阵乘 | DeepGEMM FP8 dense | `torch_npu.npu_quant_matmul`，`eager_fused`；核验 K32 block scale、UE8M0 和 FP32 输出 | M=1024/1025,N=4096,K=7168 | 无同等 FP8/scale 接口时可用 NPU `FP8→FP32→解码并应用 scale→FP32 mm`，标 `eager_composite`、说明不是 FP8 硬件吞吐对照；若 scale/精度无法对齐则 `none` |
| P07 量化/类型转换 | per_token_cast_to_fp8：SIMT/SIMD | 已有 `ref_program(x)` 对应的 NPU `pad/view→abs→amax→clamp→scale→cast→reshape/contiguous`，`eager_composite` | M=128/8192,N=8192；FP32 输入，group=128 | 当前不认定存在直接等价融合接口。若目标版本确有匹配的融合 API，可新增 `eager_fused` 行，保留组合行。若 NPU 上无法完成 FP32→E4M3FN，则 `none`，保留精度与 TL 自身基线 |
| P08 Attention forward | MHA/GQA；manual GQA 作附加配置行 | `torch_npu.npu_fusion_attention`，`eager_fused` | MHA S=512/4096,D=128；GQA S1=512/8192,G=4/32,S2=1024/8192,D=128，先验算 TL 合法分区 | 可选 NPU `torch.nn.functional.scaled_dot_product_attention`，类型记 `eager_standard`，并记录实际是否走融合实现。若只有显式 `QKᵀ→scale→softmax→PV`，单列 `eager_composite`；完整大形状放不下则无性能对照，不能缩小一边输入冒充同形状 |
| P09 Attention backward | GQA backward 完整路径 | 目标版本的 `torch_npu.npu_fusion_attention_grad`，`eager_fused` | S1=512/8192,G=2/32,S2=384/8192,D=128 | 使用 NPU SDPA 的 `torch.autograd.grad`，标 `eager_composite`（框架 backward 路径，不是单一 grad API）；预建 forward 状态，记录实际调用。若同形状或辅助状态不支持，则 `none` |

### 2.3 语义对齐与不可对照项

- **FP8 group quantization**：每 128 元素一个 scale，FP32 输入，E4M3FN 输出，
  `amax` 下限为 1e-4。已查官方 `npu_dynamic_block_quant` 文档仅列 FP16/BF16
  输入，scale 限制也不能直接当成本例 clamp，因此它只是融合候选，不能直接
  替换当前 FP32 eager 组合，也不能把普通整行 per-token quant 当成等价接口。
- **GEMM/DeepGEMM**：每条记录注明输入、累加和输出 dtype。输入 FP32 的 mm
  与输入 BF16 的 mm 不能混成同一对照。HF32 若双方舍入/精度模式无法一致，
  只保留 TL 自身性能行或明确标为跨精度探索，不计算同精度性能比。
- **RMSNorm**：Y 和 RSTD 都校验；epsilon 一致。融合入口若输出不同 shape 的
  RSTD，只有纯 view 可在计时外适配；实际计算/转换不能悄悄移出计时。
- **Grouped**：对齐 group 边界、padding、转置、bias/alpha/累加语义；不得拿
  一个 padded 大矩阵乘替代真实多组计算而不说明额外工作量。
- **Backward**：TL 计入 Delta、dQ 清零和 fused backward；双方排除 forward，
  辅助状态在计时外准备。eager autograd 需确认重复调用不重复累计梯度，不把
  forward 重算混入 backward 时间；必要的梯度清零计入对应路径。
- **DeepGEMM 辅助功能**：scale packing、非标准量化 epilogue、随机舍入等没有
  等价 eager 接口时，不强行给速度比。正确性仍按原测试完成，性能记录 TL
  自身基线。随机舍入不能用确定性 cast 当作同语义参照。

### 2.4 统一缺失处理流程

1. 先做 capability 探测：接口存在、dtype/shape 支持、实际在 NPU 执行。
2. 用小输入核对全部输出和精度模式；通过后才进入性能测量。
3. 首选接口缺失时，按上表选择明确列出的等价组合，并修改 comparison_type；
   不静默替换，记录原接口不可用原因。
4. 等价组合也不可用、同形状 OOM、输出精度无法对齐时，设置
   `comparison_type=none`，reason 使用 `api_unavailable`、`unsupported_dtype_shape`、
   `semantic_mismatch` 或 `baseline_oom`。保留 TL 正确性和 TL 耗时，不生成速度比。
5. 报告中“有有效对照/仅 TL 基线/不支持/失败”分别计数；无对照不算性能通过。

## 3. 固定对比口径

- 主表使用原有默认 block 配置，不把本次调优结果覆盖原始默认结果。
- 可附加候选集内最快配置，但作为独立 tuned 行，记录 block 和搜索范围。
- 两边同设备、shape、输入值、dtype、布局和数学功能。FP32/HF32/fast-math 精度
  模式显式记录。计时前先通过正确性。
- kernel-only（profiler 设备执行时长）和 API 端到端（包含分配/必要转换）分表。
  两边都有 out 接口时预分配；eager 无 out 接口时采用相同 profiler 口径或统一
  端到端计时，不对一边测 allocation、一边排除。
- 默认预热至少 10 次，5 个采样批，每批至少 20 次；极大 Attention 形状可降低
  次数，但双方一致。交替 A/B、B/A 顺序；独立复测至少两轮。
- warm-cache 和 cold-L2 不混表。清缓存操作必须在计时外，双方策略一致。
- eager backward/组合可能产生多个设备 kernel，完整覆盖必要计算；串行 kernel
  可求和，有并发时按依赖链或整体设备时间区间统计，不直接相加重叠时段。
  混合 AIC/AIV 并发也不能重复计算。
- 输出原始 samples、median/min/max、误差、TileLang/eager 延迟、比值
  `eager_us / tilelang_us`、comparison_type、缺失原因、版本和实际调用算子名。
  该比值大于 1 表示 TileLang 更快；组合对照只说明相对该组合的收益。
- 无历史基线时称“Torch-NPU eager 性能对比 + 首次基线建立”，不称“无版本退化”。不同类别
  不预设统一速度比通过线；后续回归在同卡同配置基线基础上判定。

## 4. 现有基础与交付状态

`eager_baseline/compare.py` 已提供 19 个可选比较项，包含标准接口、融合接口、
明确命名的组合参照。九类代表均有初版条目，但不是每个条目都具备直接融合
基线：Split-K 的 FP32 输出和 FP8 使用明确标记的组合；MHA 改选同一 builder
的 BF16 输出配置；backward 使用 SDPA autograd。具体形状、差异、失败处理和
运行命令以该目录 README 为准。当前仅完成主机侧检查，不代表 9579 已通过。

`maint/scripts/ascend_perf_regression.py` 已列出 GEMM、Split-K、DeepGEMM、scale
转换、MHA/GQA/backward、RMSNorm、VecAdd 和 FP8 quant 的 TL 性能入口，可以复用
其进程隔离与超时思路。但它目前不提供以上所有 Torch-NPU eager 对比，不能将
运行它描述为已完成 eager 性能基线。

转测交付顺序：全量正确性清单 → eager capability/语义核对 → P01/P02/P03-BF16/
P04/P08 优先对比 → grouped/FP8/backward 补充 → 归档 CSV、环境及源码。
所有类别均有指定代表；待接口对齐项保留为阻塞项，未上板项保留“未验证”。

## 官方参照

- [RMSNorm](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC1alpha003/apiref/fmkadptapi/ptaoplist_000441.html)
- [Grouped Matmul](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
- [Quant Matmul](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)
- [Dynamic Block Quant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_block_quant.md)
- [Fusion Attention](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md)
