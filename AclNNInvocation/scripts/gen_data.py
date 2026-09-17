#!/usr/bin/env python3
import argparse
import os

import numpy as np


ROWS = 8
COLS = 32


def parse_index_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "zero-link-indices must be a comma-separated list of integers"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CalcRyy inputs and golden outputs.")
    parser.add_argument("--a-num", type=int, default=40)
    parser.add_argument("--b-num", type=int, default=19)
    parser.add_argument("--max-link-per-b", type=int, default=11)
    parser.add_argument("--fixed-link-count", type=int)
    parser.add_argument("--zero-link-indices", type=parse_index_set, default=set())
    parser.add_argument("--contiguous-links", action="store_true")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--scale", type=float, default=0.125)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.a_num < 0 or args.b_num <= 0:
        raise ValueError("a-num must be non-negative and b-num must be positive")
    if args.max_link_per_b < 0 or args.max_link_per_b > args.a_num:
        raise ValueError("max-link-per-b must be in [0, a-num]")
    invalid_zero_indices = sorted(
        b_idx for b_idx in args.zero_link_indices if b_idx < 0 or b_idx >= args.b_num
    )
    if invalid_zero_indices:
        raise ValueError(
            f"zero-link-indices must be in [0, b-num), invalid: {invalid_zero_indices}"
        )
    if args.fixed_link_count is not None:
        if args.fixed_link_count < 0 or args.fixed_link_count > args.max_link_per_b:
            raise ValueError("fixed-link-count must be in [0, max-link-per-b]")
        nonzero_b_num = args.b_num - len(args.zero_link_indices)
        if args.contiguous_links and nonzero_b_num * args.fixed_link_count > args.a_num:
            raise ValueError(
                "contiguous fixed links require enough a-num rows for all nonzero B items"
            )

    rng = np.random.default_rng(args.seed)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    h_r = rng.uniform(-args.scale, args.scale, size=(args.a_num, ROWS, COLS)).astype(np.float32)
    h_i = rng.uniform(-args.scale, args.scale, size=(args.a_num, ROWS, COLS)).astype(np.float32)
    interf_power = rng.uniform(0.05, 1.5, size=(args.a_num,)).astype(np.float32)
    weak_interf = rng.uniform(0.01, 0.5, size=(args.b_num, ROWS)).astype(np.float32)

    entries = []
    contiguous_offset = 0
    for b_idx in range(args.b_num):
        if b_idx in args.zero_link_indices:
            length = 0
        elif args.fixed_link_count is not None:
            length = args.fixed_link_count
        elif args.max_link_per_b == 0:
            length = 0
        elif b_idx == 0:
            length = args.max_link_per_b
        else:
            length = int(rng.integers(0, args.max_link_per_b + 1))
        if args.contiguous_links and args.fixed_link_count is not None:
            offset = contiguous_offset
            contiguous_offset += length
            if length == 0 and args.a_num > 0:
                offset = min(offset, args.a_num - 1)
        else:
            # Even when len=0, offset is still the Hs source row. Keep it in
            # [0, aNum) whenever H exists.
            max_offset = args.a_num - max(length, 1) if args.a_num > 0 else 0
            offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
        hs_len = b_idx % (ROWS + 1)
        hs_offset = int(rng.integers(0, COLS - hs_len + 1))
        entries.extend((offset, length, hs_offset, hs_len))

    h_info = np.asarray(
        [args.a_num, args.max_link_per_b, args.b_num, *entries],
        dtype=np.uint32,
    )

    ryy_r = np.zeros((args.b_num, ROWS, ROWS), dtype=np.float32)
    ryy_i = np.zeros_like(ryy_r)
    hs_r = np.zeros_like(ryy_r)
    hs_i = np.zeros_like(ryy_r)
    for b_idx in range(args.b_num):
        offset, length, hs_offset, hs_len = entries[b_idx * 4:(b_idx + 1) * 4]
        for link in range(length):
            h_idx = offset + link
            real = h_r[h_idx] @ h_r[h_idx].T + h_i[h_idx] @ h_i[h_idx].T
            imag = h_i[h_idx] @ h_r[h_idx].T - h_r[h_idx] @ h_i[h_idx].T
            ryy_r[b_idx] += interf_power[h_idx] * real
            ryy_i[b_idx] += interf_power[h_idx] * imag
        ryy_r[b_idx, np.arange(ROWS), np.arange(ROWS)] += weak_interf[b_idx]
        if args.a_num > 0 and hs_len:
            hs_r[b_idx, :, :hs_len] = h_r[offset, :, hs_offset:hs_offset + hs_len]
            hs_i[b_idx, :, :hs_len] = h_i[offset, :, hs_offset:hs_offset + hs_len]

    h_r.tofile("input/input_h_r.bin")
    h_i.tofile("input/input_h_i.bin")
    h_info.tofile("input/input_h_info.bin")
    interf_power.tofile("input/input_interf_power.bin")
    weak_interf.tofile("input/input_weak_interf.bin")
    ryy_r.tofile("output/golden_ryy_r.bin")
    ryy_i.tofile("output/golden_ryy_i.bin")
    hs_r.tofile("output/golden_hs_r.bin")
    hs_i.tofile("output/golden_hs_i.bin")

    zero_len_indices = [
        b_idx for b_idx in range(args.b_num) if entries[b_idx * 4 + 1] == 0
    ]
    print(
        "Generate input and golden data success. "
        f"aNum={args.a_num}, bNum={args.b_num}, maxLinkPerB={args.max_link_per_b}, "
        f"zeroLenB={zero_len_indices}"
    )


if __name__ == "__main__":
    main()
