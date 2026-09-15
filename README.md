# CalcRyy AscendC custom operator

`CalcRyy` computes weighted complex covariance matrices and extracts the requested `hs` slices.
The project follows the same FrameworkLaunch layout as
`/Users/chenlj/workspace/ascendc-complex-gram/fused_complex_matmul_concat`.

## Interface

Inputs:

| name | dtype | shape |
| --- | --- | --- |
| `h_r` | `float32` | `[aNum, 8, 32]` |
| `h_i` | `float32` | `[aNum, 8, 32]` |
| `h_info` | `uint32` | `[3 + 4 * bNum]` |
| `interf_power` | `float32` | `[aNum]` |
| `weak_interf` | `float32` | `[bNum, 8]` |

`h_info` is `{aNum, maxLinkPerB, bNum, {offset, len, hsOffset, hsLen}...}`.

Outputs are four `float32` tensors with shape `[bNum, 8, 8]`:
`ryy_r`, `ryy_i`, `hs_r`, and `hs_i`.

For `H = h_r + i*h_i`:

```text
ryy[i] = sum(j=0..len-1) interf_power[offset+j] * H[offset+j] * H[offset+j]^H
diag(ryy[i]) += weak_interf[i]
hs[i, :, :hsLen] = H[offset, :, hsOffset:hsOffset+hsLen]
hs[i, :, hsLen:] = 0
```

## Build and run

On an Ascend 910B environment:

```bash
cd CalcRyyMultiCore
bash build.sh

cd ../AclNNInvocation
A_NUM=40 B_NUM=19 MAX_LINK_PER_B=11 DEVICE_ID=0 bash run.sh
```

`build.sh` builds and installs `build_out/custom_opp_<os>_<arch>.run` using the same packaging flow as the reference project. `run.sh` generates deterministic NumPy golden data, builds the generated aclnn caller, runs the operator, and verifies all four outputs.

See [DESIGN.md](DESIGN.md) for the tiling, workspace, boundary handling, and verified cases.
