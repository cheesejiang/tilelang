# UB entry 与指令组织的受控对照

从 tilelang 根目录，在具备 CANN、torch_npu 和设备权限的环境运行：

```bash
python examples/ascend/layout_workloads/run_bottleneck_study.py --output /tmp/ub-bottleneck-study
```

固定真实 `gemm_scale` 的 32x128 FP32 输入、32 threads、32 blocks，以及原来的
GM→UB→SIMT→GM 路径。默认依次运行 10 个配置，不使用独立 ASC 微基准或 volatile：

| 配置 | 目的 |
| --- | --- |
| baseline / ownership / padding-search | 复现原来的三个自动模式 |
| baseline-scalar | 相同原始布局，关闭后端向量化，与 baseline 对照 |
| p8 / p8-scalar | 固定每 8 行 +32B，分别保留/关闭后端向量化 |
| p1 / p2 / p4 / p8 / p16 | 固定 baseline 线程推导流程，扫描手工物理 padding 周期 |

每个配置单独子进程、独立目录，先验证结果，再测总耗时；随后单独采集 NPU
PipeUtilization 报告，profile 不混入 benchmark 样本。warmup 在 capture 外；capture
默认执行 10 次真实 kernel。可调整 `--profile-repeats`、`--blocks` 和 `--rounds`。
使用 `--dry-run` 仅生成计划，不导入设备模块；`--no-profile` 只进行正确性及计时。
补汇总用相同参数加 `--summarize-only`。

输出包括顶层 `summary.md/csv`，各配置的日志、逐模式 JSON、输入/推导/最终分配 IR、
生成的 `.cpp`、DMA 描述符，以及 `MODE.profile/<唯一编号>/` 下的原始 profiler 产物。
JSON 的 `profile.directory` 给出准确路径，`capture.json` 保存文件清单。
查看 profiler 导出的算子明细、PipeUtilization 指标和 trace，定位对应 VF/kernel，
比较 VECTOR/MTE 活动与等待/重叠；不同 CANN 版本可提供的字段不同。
这些利用率不是可相加的阶段耗时，不能直接用总时间乘利用率得到准确 VF/DMA 时间。
如果工具未提供足够的阶段信息，需明确保留为未确定。

分析顺序：

1. 确认各配置正确性通过，比较总耗时和多轮样本。
2. 检查 scalar/vector 成对配置的 loop ownership、地址覆盖、DMA 描述符是否一致。
   本地主机 lowering 中 baseline/p8 默认生成 float2，关闭向量化后生成标量循环；
   这仍不保证设备编译器不再合并标量。必须核对真实设备 ISA 的动态 LDS/STS、
   地址计算指令和调度，不能将 C++ 类型直接当作 ISA 宽度。
3. 比较各周期的 entry 曲线与 VF/整体时间，判断冲突收益是否出现平台；
   周期同时改变 DMA 分段，必须并看描述符和 profile，不能只用 entries 归因。
4. 相同地址的标量/向量版本若有差异，说明指令组织相关开销值得进一步分解；
   它也可能包含地址指令和调度变化，不能直接等同于固定的每条指令发射成本。

成本模型在后端向量化开关之前分析，因此 scalar 对照的模型 instructions/requests
可能仍与 vector 版一致。汇总显式标记为 model，不作为实测 ISA 数量，也不拟合
“一条指令等于多少 entry”。本实验不自动解析 ISA，不声称已经确定流水线瓶颈。
强制串行/独立度实验以及空 VF 的 DMA 对照不在这次真实 kernel 对照中。

也可单独对任意原 case 采集 profile：

```bash
python examples/ascend/layout_workloads/run_workloads.py --case gemm_scale \
  --rows 32 --columns 128 --mode all --bench --profile --output /tmp/gemm-scale-profile
```

目前完成主机测试及源码生成验证；本开发环境没有进行 NPU profiling、设备 ISA 核对或实测。
