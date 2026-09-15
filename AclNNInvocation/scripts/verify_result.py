#!/usr/bin/env python3
import argparse
import sys
from typing import Tuple

import numpy as np


ROWS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CalcRyy outputs.")
    parser.add_argument("--b-num", type=int, default=19)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    return parser.parse_args()


def check(name: str, shape: Tuple[int, int, int], atol: float, rtol: float) -> bool:
    actual = np.fromfile(f"output/output_{name}.bin", dtype=np.float32).reshape(shape)
    golden = np.fromfile(f"output/golden_{name}.bin", dtype=np.float32).reshape(shape)
    diff = np.abs(actual - golden)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(np.abs(golden), 1.0)))
    finite = bool(np.all(np.isfinite(actual)))
    ok = finite and np.allclose(actual, golden, atol=atol, rtol=rtol)
    print(f"{name}: {'PASS' if ok else 'FAIL'}, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    if not ok:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(f"  worst index={idx}, actual={actual[idx]}, golden={golden[idx]}, finite={finite}")
    return ok


def main() -> int:
    args = parse_args()
    if args.b_num <= 0:
        raise ValueError("b-num must be positive")
    shape = (args.b_num, ROWS, ROWS)
    names = ("ryy_r", "ryy_i", "hs_r", "hs_i")
    return 0 if all(check(name, shape, args.atol, args.rtol) for name in names) else 1


if __name__ == "__main__":
    sys.exit(main())
