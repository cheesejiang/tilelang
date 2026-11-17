# VecAdd SIMT blocks 对比

耗时单位为 μs。收益 = original 的 TileLang 耗时 / 当前 TileLang 耗时；S = eager 耗时 / TileLang 耗时。
28、32 表示逻辑 blocks 数，不改变设备的物理核数。

| 策略 | blocks | smoke 耗时 | smoke 收益 | smoke S | full 耗时 | full 收益 | full S |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| original | 64 | 10.730 | 1.000× | 0.664× | 9542.016 | 1.000× | 0.915× |
| auto | 56 | 9.659 | 1.111× | 0.746× | 8538.376 | 1.118× | 1.022× |
| C−1 | 55 | 10.261 | 1.046× | 0.685× | 8537.591 | 1.118× | 1.023× |
| C | 56 | 9.742 | 1.101× | 0.720× | 8560.388 | 1.115× | 1.020× |
| C+1 | 57 | 10.491 | 1.023× | 0.670× | 11417.192 | 0.836× | 0.764× |
| 2C | 112 | 10.386 | 1.033× | 0.685× | 8538.426 | 1.118× | 1.021× |
| fixed-28 | 28 | 待测 | 待测 | 待测 | 8636.992 | 1.105× | 1.012× |
| fixed-32 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

fixed-28 的 full 数据来自此前独立运行的 28 blocks 实验；收益以本表 original 耗时计算。尚无 28 blocks smoke 和 32 blocks 实测数据。

补充 VecAdd SIMT/SIMD 的默认扫描（包含原六种策略及 fixed-28、fixed-32）：

```bash
python examples/ascend/eager_baseline/block_tuning.py \
  --cases vecadd_simt vecadd_simd \
  --profiles smoke full --rounds 1 --samples 3 \
  --output vecadd-blocks-28-32
```

如只补测固定 blocks，可分别执行以下命令，并将 fixed-28 替换为 fixed-32：

```bash
python examples/ascend/eager_baseline/compare.py \
  --cases vecadd_simt vecadd_simd --profile smoke \
  --block-policy fixed-28 --rounds 1 --samples 3 \
  --output vecadd-fixed28-smoke
```

full 使用 `--profile full`，并设置不同的输出目录。以上采样参数与此前单轮、3 次采样的实验保持一致。
