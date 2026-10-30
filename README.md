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

`aNum` may be zero. In that case `h_r`, `h_i`, and `interf_power` are empty,
the Gram sum is skipped, `ryy_r` contains only `weak_interf` on its diagonal,
`ryy_i` is zero, and both `hs` outputs are zero.

When `aNum > 0`, an individual B item may still have `len=0`. Only that item's
Gram sum is skipped: its `ryy_r` is `diag(weak_interf[i])`, its `ryy_i` is zero,
and its `hs` extraction remains independent and uses `h[offset]` when
`hsLen > 0`.

`hsLen` may also be zero independently of `aNum`; no source slice is read and
both `hs` outputs are entirely zero for that B item.

## Build and run

On an Ascend 910B environment:

```bash
cd CalcRyyMultiCore
bash build.sh

cd ../AclNNInvocation
A_NUM=40 B_NUM=19 MAX_LINK_PER_B=11 DEVICE_ID=0 bash run.sh
```

`build.sh` builds and installs `build_out/custom_opp_<os>_<arch>.run` using the same packaging flow as the reference project. `run.sh` generates deterministic NumPy golden data, builds the generated aclnn caller, runs the operator, and verifies all four outputs.

For a deterministic mixed zero/nonzero-link regression, specify the zero-link B
indices while keeping a fixed link count for every other item:

```bash
A_NUM=64 B_NUM=19 MAX_LINK_PER_B=10 FIXED_LINK_COUNT=10 \
ZERO_LINK_INDICES=1,4,7,9,16 bash run.sh
```

See [DESIGN.md](DESIGN.md) for the tiling, workspace, boundary handling, and verified cases.

See [NUMERICAL_ACCURACY.md](NUMERICAL_ACCURACY.md) (中文) for numerical accuracy,
conditioning, stable downstream solves, and a proposed validation plan. These are
design recommendations, not implemented optimizations or new device benchmarks.

## CANN 8.5.1

Build the operator package with the same CANN version that will load it. In
particular, do not install a package compiled with CANN 9.x into a CANN 8.5.1
environment.

CANN 8.5.x requires NumPy 1.x. A clean virtual environment avoids accidentally
loading a system or user-site NumPy 2.x:

```bash
python3 -m venv .venv-cann-8.5.1
source .venv-cann-8.5.1/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cann-8.5.1.txt

export ASCEND_INSTALL_PATH=/path/to/Ascend/cann-8.5.1
export ASCEND_PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"

cd CalcRyyMultiCore
bash build.sh
```

`build.sh` now sources the selected toolkit, validates NumPy and the CANN Python
modules, and passes the same Python executable to every CMake/OPC step. If
`numpy.core.multiarray` is missing or NumPy 2.x is selected with CANN 8.5.x, it
stops before compilation and prints the repair command for that exact Python.

After the package is built, install the `.run` file and run the verification:

```bash
cd CalcRyyMultiCore/build_out
./custom_opp_<os>_<arch>.run

cd ../../AclNNInvocation
ASCEND_INSTALL_PATH=/path/to/Ascend/cann-8.5.1 \
ASCEND_PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python" bash run.sh
```

Validated on an Ascend 910B server with CANN 8.5.1, Python 3.12.13, and
NumPy 1.26.4. For `bNum=544` and ten links per item, 10 warmup plus 100 measured
runs passed all four output checks and measured 1.087 ms average kernel time
(1.074 ms minimum, 1.106 ms maximum on that server).

A mixed `bNum=544` case with six `len=0` items and 538 `len=10` items also
passed under CANN 8.5.1. With 10 warmups and 20 measured runs, its average
kernel time was 1.097 ms (1.061 ms minimum, 1.116 ms maximum).
