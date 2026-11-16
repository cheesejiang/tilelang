# Ascend950 多型号转测方案

## 1. 目标与范围

验证指定 Ascend950 型号在 TileLang Ascend 后端上的正确性，以及相对
Torch-NPU eager 的性能保持情况。使用 `eager_baseline/compare.py` 中的
14 个代表用例，不将结论扩展为所有算子、API 和形状完整适配。

后端为 `ascend`，执行后端为 `tvm_ffi`，编译架构为 `dav-3510`。
9579 为参照设备；新 FP8 对照用例已在 9579 复测通过，但新的 profiler
计时路径及调整后的 full 形状仍需上板验证，并重新建立性能基线。

## 2. 用例与形状

| 用例 | smoke | full |
|---|---|---|
| VecAdd SIMT/SIMD（2项） | N=2²⁰ | N=2³⁰ |
| RMSNorm fused | batch=128,d=4096 | batch=4096,d=4096 |
| GEMM BF16/FP32/mixed FP32（3项） | M=N=K=1024 | M=N=K=8192 |
| DeepGEMM BF16 dense | M=N=K=256 | M=N=K=4096 |
| DeepGEMM FP8 | M=N=K=256 | M=1024,N=4096,K=7168 |
| DeepGEMM BMM | batch=2,M=N=K=256 | batch=8,M=N=K=1024 |
| DeepGEMM grouped | groups=2,每组M=N=K=256 | groups=8,每组M=N=K=1024 |
| MHA forward | S=512,D=128 | S=4096,D=128 |
| GQA forward/manual/backward（3项） | S1=128,G=32,S2=1024,D=128 | S1=8192,G=32,S2=8192,D=128 |

N 在 VecAdd 中为元素数；M/N/K 为矩阵乘维度；S 为序列长度；GQA 的
S1/G/S2/D 分别为 query 序列长度、共享 KV 的 query head 数、KV 序列长度、head 维度。

full 对齐原始性能入口；DeepGEMM dense 选取原性能回归清单中的代表形状。
BMM/grouped 无固定原始性能入口，保留上表代表形状。
smoke 和 full 分别报告，各型号必须比较同一形状。

## 3. 对照与计时方法

对照使用 eager 模式直接调用的 `torch`/`torch_npu` 接口，不使用
`torch.compile` 或图编译。主要包括 add、mm、bmm、npu_rms_norm、
npu_grouped_matmul、npu_quant_matmul、npu_fusion_attention。
GQA backward 当前对照为 NPU SDPA + autograd.grad，并非直接 grad 融合接口。
具体接口归属以结果中的 eager_api/comparison_type 为准，不根据接口名推定
底层来自 ops-nn 或 Ascend C。

每个用例按以下顺序执行：

1. 双方运行并完成既定数值校验，失败则不计时。
2. 双方各预热3次，每次前执行1 GiB缓存驱逐并同步。
3. 双方各正式采样15次，按采样序号交替先后顺序。
4. 每次采样前缓存驱逐并同步，随后在独立profiler窗口内执行一次完整调用。
5. 统计窗口内所有有效kernel执行时间区间的并集，不重复计算重叠执行时间。
6. 计时后再次运行并检查正确性，保存原始数据与统计结果。

清缓存、编译、输入准备和校验不计时；必要的设备计算、转换和清零保留在
被测调用内。输出分配仍在调用内，但Python和主机分配开销不属于设备busy指标。
以15次设备耗时的算术平均值计算加速比，同时保存median/min/max/stdev。
该方法参考TileOPs Ascend计时协议，但未引入其预约与锁频管理。
应保证设备空闲、功耗/频率策略一致，并记录软件及硬件环境。
缓存驱逐不影响算子内部的数据复用，也不保证所有型号的每级缓存绝对清空。

profiler缺少有效记录、清缓存混入计时窗口等情况判为失败，不自动退回event。
旧event/wall结果不得与新的profiler基线混用。

## 4. 执行方式

在已构建的TileLang仓库根目录执行。脚本默认三轮；以下命令显式指定轮数。

快速检查：

```bash
python examples/ascend/eager_baseline/compare.py \
  --profile smoke --timing profiler --rounds 1 --output eager-smoke
```

9579正式参照：独立运行三轮，确认基线稳定。

```bash
python examples/ascend/eager_baseline/compare.py \
  --profile full --timing profiler --rounds 3 --output eager-9579-full
```

其他型号：先运行一轮；波动明显、保持率接近或低于验收线时，再独立补测两轮。
补测必须全部纳入报告，不丢弃慢轮次；必要时重新完整执行三轮并保留前次结果。

```bash
python examples/ascend/eager_baseline/compare.py \
  --profile full --timing profiler --rounds 1 --output eager-target-full
```

每个用例独立子进程，默认超时900秒（包括编译、校验和profiler采集）。
超时后如需调整，应记录原因并使用一致的测试策略重新运行。

## 5. 验收指标与标准

```text
加速比 S = TileLang性能 / Torch eager性能
         = eager平均设备耗时 / TileLang平均设备耗时

相对保持率 R = 待测型号的S / 9579的S
```

三轮汇总对各轮平均耗时取算术平均，再计算双方耗时之比。
单轮结果使用该轮15次采样的均值。当前脚本输出S，跨型号R需另行汇总。

| 检查项 | 标准 |
|---|---|
| 编译与执行 | 14个用例均正常完成，无编译错误、设备异常、超时 |
| 正确性 | 全部通过脚本既定的数值校验 |
| 有效性能数据 | 双方均有有效profiler记录及完整采样 |
| 相对性能保持 | 建议验收线为逐项R≥90%，转测启动前确认；低于或接近阈值时复测确认 |
| 结果完整性 | 环境、日志、原始数据与汇总齐全；失败、缺失、对照不可用均不计通过 |

不要求S>1；原先慢于eager的算子也可以保持其相对性能。
逐项判定，不用平均加速比掩盖单项下降；波动过大时暂不判定。
任一必测轮失败或对照不可用，该项不生成有效的三轮汇总加速比。

本范围不额外承诺Split-K确定性、所有非整除尾轮、全部dtype/API等特殊路径覆盖。
性能保持率衡量相对eager的表现，不单独证明TileLang无版本退化；版本回归需同卡比较改前/改后。

## 6. 结果归档

单轮（`--rounds 1`）：输出目录直接包含environment.json、results.csv、
各case的JSON和日志，以及runs下的源码与profiler数据。

三轮：

```text
输出目录/
├── results.csv                  # 全部逐轮结果
├── summary.csv                  # 三轮汇总
├── rounds.json                  # 参数与汇总
└── round-XX-时间戳/
    ├── environment.json
    ├── results.csv
    ├── <case>.json
    ├── <case>.log
    └── runs/<case>_时间戳/profiles/
        ├── flush-check/
        ├── tilelang/repeat-00/...
        └── eager/repeat-00/...
```

每次采样耗时查看case JSON的tilelang_samples_us/eager_samples_us（单位μs）；
解析明细查看profiler_records。repeat目录的PROF子目录下ffts_profile*为
原始二进制文件，实际层级取决于CANN版本；解析器依赖TileOPs采用的CANN9.2 FFTS格式。

汇总表记录型号、用例、形状、dtype、双方平均耗时、S、R及异常说明。
验收通过结论写为：该型号通过所选14个代表用例的正确性与相对性能验收。
