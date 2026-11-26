# Ascend950 多机型兼容与验收说明

本轮沿用现有 TileLang Ascend 后端，无需为目标机型修改接口或代码。兼容性分析有两点。

1. **从 DSL 到内核提交，已核对的流程不按 SKU 后缀筛选。** 针对 TVM FFI 链路，TileLang 根据内核的操作、数据类型和内存域等信息生成 Ascend C 源码，Bisheng 按指令架构编译、链接出设备二进制。执行时，TVM FFI 通过 CANN 在当前设备上加载二进制。这些步骤均无需匹配 `9579`、`9592` 等型号后缀。
2. **TileLang 调用 Bisheng 时，目标机型共用 `dav-3510` 编译架构。** `get_target_npu_arch()` 优先取编译目标的 `arch` 或 `mcpu`，其次取环境变量 `ASCEND_NPU_ARCH`，默认值为 `dav-3510`，传给 Bisheng 的参数是 `--npu-arch=dav-3510`。已对照的目标机型规格中，`NpuArch` 均为 `3510`，核内存储容量也一致。UB 为 **253,952 字节**、L1 为 524,288 字节、L0A 和 L0B 各为 65,536 字节、L0C 为 262,144 字节。因此这些机型可以沿用同一架构和核内存储上限编译。核数、L2 和整卡显存以实际 SKU 规格为准。

**结论。** 从现有后端接口、默认编译架构和已核对的核内存储规格看，目标机型具备兼容基础。实机运行需要匹配的驱动、运行时和 CANN。已有 `Ascend950PR_9579` 上板反馈和 `Ascend950DT_9592` 宿主机正确性通过记录。

**验收。** 每个目标 SKU 自选可复现的正确性测试。真机编译、执行和精度检查通过即可，记录实际机型、环境、结果和日志。

## 950DT 自测报告

2026-11-18 在 950DT 宿主机验证，设备为 `Ascend950DT_9592`，上报 72 个 AIV。TileOPs 使用 `ascend950-v0.1.0` 分支的 `393a84eb` 提交。

| 自测内容 | 结果 | 950DT 原始日志 |
|---|---|---|
| Paged cache gather | 自动选 72、固定 64、固定 80，各 4 项通过。与 CPU 参考逐元素一致。 | `/home/jyw/gather-results/paged-gather-blocks-20261118.log` |
| RoPE、Softmax、融合 RMSNorm 量化 | CANN 9.2.0-beta.2 下分别通过 6、6、3 项。同轮两项 BMM 在编译阶段失败。 | `/home/jyw/gather-results/ops-ascend-20261118-073401.log` |
| BMM FP16、BF16 | 切到 CANN 9.3 后，两项均完成编译、真机执行及精度比对，pytest 报告 `2 passed in 45.33s`。 | `/home/jyw/gather-results/bmm-cann93-20261118.log` |

### 宿主机复跑命令

先执行 `ssh 950DT`，再在宿主机 Bash 中定义以下函数。它复用已验证的 CI Python 和设备组配置，整个测试在共享锁内执行。复跑日志使用新文件名。

```bash
run_950dt() {
  local suite="$1" log
  log="/home/jyw/gather-results/${suite}-$(date +%Y%m%d-%H%M%S).log"
  (
    set -o pipefail
    sudo -n /usr/local/bin/npu-guard run \
      --owner "jyw/$suite" --wait 3600 --timeout 3600 \
      -- /usr/sbin/runuser -u jyw -g jyw -G HwHiAiUser -- /bin/bash -c '
        set -e
        cd /home/jyw/workspace/TileOPs
        if [[ "$1" == bmm-cann93 ]]; then
          sdk=/home/jyw/gather-cache/cann93-host.lt1tQL/cann-9.3.0
          test -r "$sdk/set_env.sh"
          source <(sed "s|/opt/Ascend/cann-9.3.0|$sdk|g" "$sdk/set_env.sh")
          export BISHENG_HOME="$sdk/x86_64-linux" PATH="$sdk/x86_64-linux/bin:$PATH"
          export LD_LIBRARY_PATH="$sdk/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          command -v bisheng
          bisheng --version
        else
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
        fi
        export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        py=/opt/tileops-ci/runtime/venv/bin/python
        suite="$1"
        shift
        if [[ "$suite" == paged-gather ]]; then
          script=/home/jyw/gather-results/tileops-950dt-gather-check.py
          "$py" -u "$script"
          "$py" -u "$script" --blocks 64
          exec "$py" -u "$script" --blocks 80
        fi
        exec "$py" -u -c \
          "import torch_npu, tileops._kernels.ascend, pytest; raise SystemExit(pytest.main())" \
          -v --tb=short -p no:cacheprovider --tileops-device=npu:0 --tileops-target=detect "$@"
      ' bash "$suite" "${@:2}" 2>&1 | tee "$log"
  )
}
```

```bash
run_950dt paged-gather

run_950dt vector-ops -m smoke \
  tests/ops/test_rope.py::test_rope_1d \
  tests/ops/test_softmax.py::test_softmax_op \
  tests/ops/test_fused_add_rms_norm_per_token_quant.py::test_fused_add_rms_norm_per_token_quant

run_950dt bmm-cann93 \
  'tests/ops/test_bmm.py::test_bmm[smoke-fp16-b4-128]' \
  'tests/ops/test_bmm.py::test_bmm[smoke-bf16-b4-128]'
```

这些命令整理自实测调用。向量算子复跑只选原日志中已通过的用例。报告适用于 9592 宿主机及表中工具链。

### 950PR 与 950DT 的 BF16 GEMM 同输入对照

2026-11-19，两台机器使用同一份 `8192 × 8192` BF16 矩阵 A、B，执行 BF16 GEMM，关闭累加，分别记录 BF16 和 FP32 输出。输入来自种子 `9592` 的独立 BF16 诊断用例。

- 输入 SHA256：`201d177ae86e3c73810ebe3206f0ad103eb0ccd6cf123a5a629cab87e47e0fdb`。 [GitCode 输入与校验文件](https://gitcode.com/qq_50709267/tmp/tree/bf16-gemm-fixture)
- 950PR_9579 使用 TileLang `9330dae8`、PyTorch `2.10.0+cpu`、`torch_npu 2.10.0.post2`，结果来自用户提供的实机输出。
- 950DT_9592 使用 TileLang `d60ec7a4`、PyTorch `2.13.0+cpu`、`torch_npu 2.13.0.rc1`、CANN `9.3.0`。原始日志位于 950DT 的 `/home/jyw/gather-results/bf16-gemm-saved-inputs-950dt.log`。

#### 测试方法

1. 从 GitCode 下载 `bf16-gemm-inputs-seed9592.pt`，用同分支的 `.sha256` 文件校验。两台机器直接读取保存的 BF16 张量，输入不依赖本机随机数生成器。
2. 从 TileLang 仓库根目录运行 `python /path/to/tilelang-bf16-gemm-compare-inputs.py /path/to/bf16-gemm-inputs-seed9592.pt`。脚本在 950DT 上位于 `/home/jyw/gather-results/tilelang-bf16-gemm-compare-inputs.py`。在 950DT 上仍按共享卡约定在 `npu-guard` 内执行。
3. 在每台机器上运行四条路径。TileLang 使用同一 `gemm()` 构造函数和 `MIXED=False`，分别编译成 BF16 和 FP32 输出。PyTorch BF16 参考使用 `a @ b.T`，FP32 参考使用 `a.float() @ b.float().T`。把各自的输出搬回 CPU，对连续张量的原始字节计算 SHA256，并记录逐元素误差。

BF16 诊断是单独运行的用例。完整 `example_gemm.py --no-bench` 即使同样设种子 `9592`，也会先运行前三组并消耗随机数，所以第四组使用的是另一份输入。950DT 完整示例的固定种子复跑最大绝对误差为 1.0，日志为 `/home/jyw/gather-results/example-gemm-seed9592-20261119.log`。本节的 2.0 来自两机共用的保存输入。

#### 跨机器输出校验

| 输出路径 | 950PR_9579 SHA256 | 950DT_9592 SHA256 |
|---|---|---|
| TileLang BF16 | `a332b4c42fc490ea2b18a2afebae1e2584f64f7da05a29b2a574724aee93b799` | 同 PR |
| TileLang FP32 | `4d898299241b4e290e52b74c4662aa1635ebd3c5d7c40a0c618d78e27e638d9a` | 同 PR |
| PyTorch BF16 | `a332b4c42fc490ea2b18a2afebae1e2584f64f7da05a29b2a574724aee93b799` | `78dc0beddbfdfdb4bb252aceff6dfd1d174bdf7e1821f368220264cb88d0c2f3` |
| PyTorch FP32 | `05286cebc51ba6a19742d03fd3d864b220985933622313e2d1da746858e55d9f` | `f837160a74e44e27b336070ced96c959cd3e354ddf24ec96f9266d08e23114ba` |

#### 逐元素误差

每份输出有 67,108,864 个元素。「不相等」按逐元素精确比较计算。「超限」按 `|实际值 - 参考值| > 0.01 + 0.01 × |参考值|` 计算。

| 同机比较 | PR 最大绝对误差／不相等元素数／超限数 | DT 最大绝对误差／不相等元素数／超限数 |
|---|---|---|
| TileLang BF16 与 PyTorch BF16 | 0／0／0 | 2.0／301／0 |
| TileLang BF16 与 TileLang FP32 转 BF16 | 0／0／0 | 0／0／0 |
| TileLang BF16 与 PyTorch FP32 转 BF16 | 2.0／27,822／0 | 2.0／27,828／0 |
| TileLang FP32 与 PyTorch FP32 | 0.000732421875／63,036,303／0 | 0.000732421875／63,036,172／0 |

在 DT 的 301 个 BF16 不相等元素中，有 20 个相差超过一个相邻 BF16 档位，所有元素仍满足上表的容差。两机的 TileLang BF16 输出均与自身 FP32 输出转成 BF16 后的结果逐元素一致。

#### 最大误差位置与验收判定

DT 的最大误差位置为零基索引 `(7264, 771)`。用同一行 A 和同一行 B 在 CPU 上重算 FP64 点积，得到 `-260.99998156`，直接舍入到 BF16 得到 `-260`。TileLang FP32 输出为 `-261.00018311`，DT 的 PyTorch FP32 参考为 `-260.99993896`，两者相差约 0.00024，位于 BF16 舍入中点 `-261` 两侧。两机的 TileLang BF16 均为 `-262`，PR 的 PyTorch BF16 为 `-262`，DT 的 PyTorch BF16 为 `-260`。PR 在这个位置的 PyTorch FP32 值未单独记录。

- **原示例规则。** `example_gemm.py` 要求全矩阵 `max_diff < 0.01`。对这份输入，PR 的 BF16 误差为 0，满足阈值。DT 的误差为 2.0，超过阈值。
- **pytest 容差。** 将 `test_gemm.py` 对 BF16 输出使用的 `rtol=0.01, atol=0.01` 应用到本次大矩阵对照。此位置允许误差为 `0.01 + 0.01 × 260 = 2.61`，实际误差为 2.0。两张卡全矩阵均无超限元素。这里记录的是诊断脚本按相同容差计算的结果，原示例的断言失败单独保留。
- **结论范围。** 这份输入在两机上的 TileLang BF16 和 FP32 输出分别逐字节一致，原示例阈值的判定差异来自两机的 PyTorch BF16 参考值。FP64 点积只用于解释所抽查的最大误差位置。两机使用的 TileLang、PyTorch 和 `torch_npu` 版本不同，尚未隔离哪项环境差异导致参考值变化。
