# CalcRyy aclnn invocation

This directory generates inputs and NumPy golden outputs, builds a C++ caller, invokes the installed custom operator through the generated two-stage aclnn API, and verifies all four outputs.

```cpp
aclnnCalcRyyGetWorkspaceSize(..., &workspaceSize, &executor);
aclnnCalcRyy(workspace, workspaceSize, executor, stream);
```

Build and install `../CalcRyyMultiCore` first, then run:

```bash
A_NUM=40 B_NUM=19 MAX_LINK_PER_B=11 DEVICE_ID=0 bash run.sh
```

Available environment variables are `A_NUM`, `B_NUM`, `MAX_LINK_PER_B`, `DEVICE_ID`, `WARMUP_COUNT`, `REPEAT_COUNT`, and `CHECK_EACH_RUN`. Set `FIXED_LINK_COUNT` to force every B item to use the same `len`; with `CONTIGUOUS_LINKS=1`, their link ranges are laid out consecutively and require `A_NUM >= B_NUM * FIXED_LINK_COUNT`.

For a repeatability check:

```bash
A_NUM=40 B_NUM=19 MAX_LINK_PER_B=11 \
WARMUP_COUNT=2 REPEAT_COUNT=10 CHECK_EACH_RUN=1 bash run.sh
```

For the `bNum=544`, average `len=10` performance case:

```bash
A_NUM=5440 B_NUM=544 MAX_LINK_PER_B=10 \
FIXED_LINK_COUNT=10 CONTIGUOUS_LINKS=1 \
WARMUP_COUNT=20 REPEAT_COUNT=500 CHECK_EACH_RUN=0 bash run.sh
```

With `CHECK_EACH_RUN=1`, the caller poisons output buffers before each measured launch and verifies that every repeated result has the same hash. The final result is always compared with NumPy golden files.
