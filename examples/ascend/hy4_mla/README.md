# Hy4 BF16 sparse MLA on Ascend

Sparse MLA for Hy4 on Ascend 950. Scores cover 512 latent dimensions and
64 RoPE dimensions, and the kernel writes a 512-dimensional latent output.
The caller supplies the physical KV row indices.

| Files | Purpose |
| --- | --- |
| `kernel.py` | H64 BF16 core, QK576 to V512 |
| `compile_sparse_fwd.py` | Compilation entry point |
| `interface.py` | BF16 call adapter |
| `testing.py` | Inputs, the FP32 reference, and numerical checks |
| `run.py` | Correctness check and optional warm-cache timing |
| `test_hy4_mla.py` | NPU regressions for the core |
| `test_interface.py`, `test_interface_npu.py` | CPU and NPU tests of the call adapter |

```python
from examples.ascend.hy4_mla.interface import flash_mla_sparse_fwd

output, maximum, lse = flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale,
    attn_sink=attn_sink,
    topk_length=topk_length,
)
```

| Argument | Shape | Type |
| --- | --- | --- |
| `q` | `[N, H, 576]` | BF16 |
| `kv` | `[M, 1, 576]` | BF16 |
| `indices` | `[N, 1, K]` | int32 |
| `attn_sink` | optional `[H]` | FP32 |
| `topk_length` | optional `[N]` | int32 |
| `output` | `[N, H, 512]` | BF16 |
| `maximum`, `lse` | `[N, H]` | FP32 |

`K` is a positive multiple of 64. `H` is from 1 to 64; the kernel computes 64
heads, and the adapter pads a smaller head count before returning the active
heads. `sm_scale` is supplied by the caller. The caller applies request and
causal filtering when constructing the indices. Negative or out-of-range
indices and slots beyond the optional length prefix are masked. Sink
contributes to output normalization. The returned statistics cover the selected
KV scores. Empty rows return zero output, negative-infinite maximum, and
positive-infinite LSE. With H=64, a caller-provided `out` is bound directly.
With fewer heads, the adapter copies the active heads out of a padded buffer.

On an Ascend 950PR with 28 AICs and CANN 9.2.0 B112, this kernel reaches about
**365 TFLOPS** at N=4096, M=4096, H=64, TopK=2048, QK=576, and V=512. The numerator
counts only the QK and PV matrix multiplications: `2×576 + 2×512` FLOPs per TopK slot per head.

## Run

Build TileLang with Ascend support; see the [backend guide](../../../tilelang/ascend/README.md).

From the TileLang repository root, select an idle NPU with
`ASCEND_RT_VISIBLE_DEVICES`, then run:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with an idle physical NPU.
export PYTHONPATH=.
python -m pytest examples/ascend/hy4_mla -q
python -m examples.ascend.hy4_mla.run --nq 4096 --nk 4096 --topk 2048 --repeat 10
```

`test_interface.py` runs on CPU. The NPU tests skip when no Ascend device is
available. `run.py` checks an independent FP32 reference first. `--repeat`
enables warm-cache NPU Event timing and prints JSON; the default is a
correctness check only. `--no-pipeline` disables the pipelined schedule, and
`--num-cores` overrides the launch core count. The TFLOPS figure above is a
separate `msprof op` measurement. `run.py` does not flush L2 or report it.

Pair gathers use a zero-or-two burst count. The local `T.assume(n_burst > 0)`
suppresses the compiler's copy guard while retaining the runtime DMA count;
it is a code-generation workaround, not a valid input invariant. A zero
burst must preserve the UB contents and issue no GM read.
