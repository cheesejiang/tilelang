# Ascend 编译阶段的 blocks 自动分配

普通 `ascend` 后端在启动结构具体化之后、任务调度、同步插入和 host/device
拆分之前，执行公共编译 Pass `tl.ascend.AutoBlocks`。不要求现有算子的 DSL
增加参数。PTO 后端不启用此 Pass。

对于满足条件的独立任务循环，选择规则为：

```text
blocks = min(执行资源的核数, 任务 tile 总数)
```

包含 Cube GEMM 的 kernel 使用 AIC 核数，包括由自动调度生成 Cube/Vector
协同 epilogue 的情况。纯 SIMD/SIMT VF kernel 使用 AIV 核数。
无法识别执行模式时，保持原配置。

优先读取当前选中 NPU 的核数；某类资源的核数读取不到时，使用手动环境变量
`TILELANG_ASCEND_AIC_CORES` 或 `TILELANG_ASCEND_AIV_CORES` 作为回退值。
手动值必须为正整数。既无法读取、又没有手动值时，不对该类资源进行优化。
当前约定：在当前设备上编译，并在当前设备上运行。

```bash
# 无卡编译时使用手动回退值；能读取设备核数时，设备值优先。
TILELANG_ASCEND_AIC_CORES=28 TILELANG_ASCEND_AIV_CORES=56 python your_program.py

# 关闭优化，保留原始 grid，用于诊断或优化前后对比。
TILELANG_ASCEND_AUTO_BLOCKS=0 python your_program.py
```

在查询完整 kernel 编译缓存之前，将核数配置固定到 PrimFunc 属性中，
使其参与缓存标识。设备核数或优化开关变化时，相关 JIT 内存缓存会失效。
直接进行分阶段 lowering 时，在进入该 Pass 时解析同一套规则。

## 适用条件与跳过规则

Pass 识别一维 `T.Persistent` 任务域，以及完整任务索引为
`step * original_blocks + bx` 的标准 grid-stride 循环。
改写时同时更新任务索引、循环轮数和 grid。
如果原本整除的任务分配在改写后出现非整除尾轮，则为完整任务体添加边界保护。

改写要求能够识别任务的输出归属，不存在同一全局 buffer 的反馈读写，
且临时状态在每个任务内完成初始化。
没有 `T.Persistent` 标记的向量循环，必须写入互不重叠的一维任务区域。
存在显式同步、原子操作、多维或显式混合启动、含义不明确的循环结构，
或无法确认的任务状态时，跳过优化。

当前 Attention forward 这类按连续区间均分任务的调度方式不做转换。
无法识别任务循环的固定 grid、动态任务边界，以及 DeepGEMM 自有调度器均保持原样。
本实现不搜索合法因数，不进行离线调优，也不使用性能模型，不保证配置达到最优性能。

eager blocks 扫描脚本会在候选 PrimFunc 上显式关闭此公共 Pass，
避免覆盖实验指定的 blocks，保留 `original` 基线和显式 blocks 实验的含义。

## 验证方式

仅依赖主机的 IR 测试（需要使用启用了 Ascend 后端的 TileLang 构建）：

```bash
python -m pytest testing/ascend/transform/test_ascend_auto_blocks.py -q
```

在 NPU 上，通过原有正确性测试入口验证公共 Pass：

```bash
TILELANG_DEFAULT_TARGET=ascend TILELANG_EXECUTION_BACKEND=tvm_ffi \
  python -m pytest examples/ascend/test_vecadd.py examples/ascend/test_gemm.py -v
```

进行优化前后对比时，使用 `TILELANG_ASCEND_AUTO_BLOCKS=0` 恢复原始编译行为。
主机侧的 IR 任务覆盖检查和生成代码检查，不能代替 NPU 上的数值正确性验证。

## 全部性能用例的优化前后验证

在 9579 等实际 NPU 环境中执行以下命令，覆盖 `eager_baseline` 已选定的全部
14 个正确性及性能用例，包括 VecAdd SIMT/SIMD、RMSNorm、三种普通 GEMM、
四种 DeepGEMM、MHA、GQA、GQA manual 和 GQA backward：

```bash
python examples/ascend/eager_baseline/compiler_blocks.py \
  --profiles smoke full \
  --rounds 3 --warmup 3 --samples 15 \
  --output compiler-blocks-all
```

脚本为每个用例、每种规格分别启动关闭/开启公共 Pass 的独立测试进程，
并打乱任务顺序。两侧均保留原始 DSL 配置，不通过 Python 预先选择优化后的 blocks。
未被 Pass 改写的算子也参与测试，用于检查数值正确性和性能变化。

每轮先检查完整输出的数值正确性，再预热 3 次，用 Profiler 正式采样 15 次，
共独立执行 3 轮。沿用现有计时协议：每次调用前执行 1 GiB 缓存驱逐并同步，
统计设备忙碌时间区间的并集。编译和输入准备不计入耗时。
为确认实际执行了 Pass，此验证模式关闭完整 kernel 编译缓存；编译记录中保存
各 kernel 改写前后的 blocks。该操作不会关闭正常应用中的编译缓存。

输出目录包含：

| 文件 | 内容 |
| :--- | :--- |
| `summary.md` | 按 smoke/full 展示所有用例的实际改写情况、两侧耗时、收益和 S |
| `summary.csv` | 同一组对比结果，便于进一步分析 |
| `results.csv` | 各关闭/开启测试任务的原始结果及结果目录 |
| `experiment.json` | 测试参数和实际任务顺序 |
| `runs/` | 各测试任务的日志、采样数据及编译记录；编译记录位于对应 worker 目录下的 `compiler-blocks.jsonl` |

指标定义：

- 优化收益 = 关闭 Pass 的 TileLang 耗时 / 开启 Pass 的 TileLang 耗时。
- S = 同一侧 eager 耗时 / 同一侧 TileLang 耗时。
- `compiler_changed=yes` 表示编译记录中确实观察到 grid 改写；`no` 表示观察到 Pass，
  但没有改写；`unknown` 表示未获得编译记录，不等同于未改写。
- 多 kernel 用例的 blocks 可以是列表，不将其误记为单一 kernel 的启动规模。

两侧正确性均通过且设备、形状、精度和计时方式一致，才计算收益。
收益大于 1 表示更快，小于 1 表示更慢；是否达到性能验收标准需按约定阈值判断。
失败、不支持、缺失和无法比较项单独记录，不计为通过。

如先进行快速检查，可使用 `--rounds 1 --samples 3`；正式结论使用上面的完整参数。
