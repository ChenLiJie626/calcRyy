# CalcRyy operator design

## 1. Semantics

The complex tensor is represented by two `float32` tensors:

```text
H = h_r + i * h_i, shape [aNum, 8, 32]
```

`h_info` has length `3 + 4*bNum`:

```text
h_info = {
  aNum,
  maxLinkPerB,
  bNum,
  {offset, len, hsOffset, hsLen} for every B item
}
```

For each B item:

```text
Ryy = 0
for link in [0, len):
  h = H[offset + link]
  Ryy += interf_power[offset + link] * h * h^H
for row in [0, 8):
  Ryy[row, row] += weak_interf[b, row]

Hs[:, :hsLen] = H[offset, :, hsOffset:hsOffset + hsLen]
Hs[:, hsLen:] = 0
```

Writing `h = Hr + i*Hi`, one link contributes:

```text
real(h*h^H) = Hr*Hr^T + Hi*Hi^T
imag(h*h^H) = Hi*Hr^T - Hr*Hi^T
```

## 2. Operator interface

| index | input | dtype | shape |
| --- | --- | --- | --- |
| 0 | `h_r` | `float32` | `[aNum, 8, 32]` |
| 1 | `h_i` | `float32` | `[aNum, 8, 32]` |
| 2 | `h_info` | `uint32` | `[3 + 4*bNum]` |
| 3 | `interf_power` | `float32` | `[aNum]` |
| 4 | `weak_interf` | `float32` | `[bNum, 8]` |

| index | output | dtype | shape |
| --- | --- | --- | --- |
| 0 | `ryy_r` | `float32` | `[bNum, 8, 8]` |
| 1 | `ryy_i` | `float32` | `[bNum, 8, 8]` |
| 2 | `hs_r` | `float32` | `[bNum, 8, 8]` |
| 3 | `hs_i` | `float32` | `[bNum, 8, 8]` |

`InferShape` derives all output shapes from `weak_interf.shape[0]` and verifies that `h_info.shape[0] == 3 + 4*bNum`. `aNum` and `bNum` are also saved in tiling data so the kernel can reject a mismatch between tensor shapes and the runtime `h_info` header.

## 3. Real-GEMM conversion

Each GEMM combines up to eight links. Its fixed logical shape is:

```text
M = 16
N = 16
K = 8 links * 2 complex planes * 32 = 512
```

Only the first eight N columns carry data; N is padded to 16 for Cube alignment.
For every active link, the AIV side packs:

```text
A = [p*Hr,  p*Hi]   shape [16, 512]
    [p*Hi, -p*Hr]

B = [Hr, Hi]         physical shape [16, 512]
                     rows 8:16 are zero
```

The AIC side calls Matmul with `transposeB=true`:

```text
C = A * B^T, shape [16, 16]
C[0:8, 0:8]  = weighted real Gram contribution
C[8:16, 0:8] = weighted imaginary Gram contribution
```

If `maxLinkPerB > 8`, the kernel iterates over `ceil(maxLinkPerB/8)` link chunks and accumulates the valid C regions in workspace.

## 4. Parallel scheduling and synchronization

`blockDim=16`. Each AIC block owns one workspace slot and one B item in the current wave. With the 910B mixed AIC/AIV task ratio, two AIV blocks cooperate with each AIC slot and split the eight antenna rows.

```text
slotIdx  = AIC block index
bIdx     = groupBase + slotIdx
groupBase advances by 16
```

For each link chunk:

1. AIV zeroes and packs A/B for active B items.
2. All cores synchronize.
3. AIC runs the fixed `16 x 512` by `512 x 16` GEMM.
4. All cores synchronize.
5. AIV adds the first eight C columns to the per-slot accumulator.

After the last chunk, AIV writes `ryy`, adds `weak_interf` to the real diagonal, and writes the zero-padded `hs`. A final wave barrier prevents workspace slots from being reused while another core still writes the preceding wave.

## 5. Workspace

Each of 16 concurrent slots contains:

| block | elements | bytes |
| --- | ---: | ---: |
| A | `16 * 512 = 8192` | 32768 |
| B | `16 * 512 = 8192` | 32768 |
| C | `16 * 16 = 256` | 1024 |
| accumulator | `16 * 8 = 128` | 512 |

User workspace is `16 * 67072 = 1,073,152` bytes. The host adds `GetLibApiWorkSpaceSize()` for Matmul. On `DevEnv_736084`, the aclnn call reported a total workspace of `17,850,368` bytes.

## 6. Boundary behavior

- Tensor shapes require positive `aNum` and `bNum`.
- Runtime `h_info[0]` and `h_info[2]` must equal the shape-derived `aNum` and `bNum`; otherwise the kernel returns without writing outputs.
- `maxLinkPerB` is bounded by `aNum` in the kernel.
- Each `len` is bounded by `maxLinkPerB` and by the remaining input range `aNum-offset`.
- Invalid `offset` produces no Gram contribution and a zero `hs`.
- `hsLen` is bounded to 8 and to the remaining source columns after `hsOffset`; invalid `hsOffset` produces a zero `hs`.
- `weak_interf` is added only to the real diagonal.

## 7. FrameworkLaunch layout

The project follows the requested reference layout:

```text
calcRyy/
|-- AclNNInvocation/
|   |-- inc/
|   |-- scripts/gen_data.py
|   |-- scripts/verify_result.py
|   `-- src/
|-- CalcRyyMultiCore/
|   |-- cmake/
|   |-- op_host/calc_ryy.cpp
|   |-- op_host/calc_ryy_tiling.h
|   |-- op_kernel/calc_ryy.cpp
|   `-- build.sh
`-- CalcRyy.json
```

The generated API is:

```cpp
aclnnCalcRyyGetWorkspaceSize(...);
aclnnCalcRyy(...);
```

## 8. Device verification

The operator package and aclnn caller were built and run on Ascend server `DevEnv_736084` (Ascend 910B2). Verified cases:

| aNum | bNum | maxLinkPerB | purpose |
| ---: | ---: | ---: | --- |
| 1 | 1 | 1 | minimum non-empty case |
| 9 | 5 | 0 | zero-link path |
| 24 | 16 | 8 | exact link-chunk and B-wave boundary |
| 32 | 17 | 9 | both boundaries crossed by one |
| 40 | 19 | 11 | default mixed case and repeated execution |
| 80 | 33 | 25 | four link chunks and three B waves |

All four outputs passed NumPy comparison. The largest observed absolute error across these runs was `1.91e-6`; both `hs` outputs matched exactly. A `2`-warmup, `3`-repeat run also passed every output-hash consistency check.
