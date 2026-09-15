# CalcRyyMultiCore

This directory is the custom-operator project for `CalcRyy`, following the Ascend `MatmulCustomMultiCore` operator-project layout.

- `op_host/calc_ryy.cpp`: OpDef registration, shape checks, tiling generation, blockDim, tiling key, and workspace size.
- `op_host/calc_ryy_tiling.h`: tiling data definition registered as `CalcRyy`.
- `op_kernel/calc_ryy.cpp`: AscendC kernel entry `calc_ryy`.

The kernel processes 16 B items per wave. Each AIC/AIV workspace slot packs up to eight links into one `M=16, N=16, K=512` real GEMM, accumulates additional link chunks, adds the real diagonal term, and writes the zero-padded `hs` slice.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before running `../AclNNInvocation/run.sh`.
