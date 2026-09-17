#!/usr/bin/env python3
"""Validate the Python environment used by the CANN operator toolchain."""

import argparse
import importlib
import pathlib
import re
import sys
from typing import Optional


def read_cann_version(cann_path: pathlib.Path) -> Optional[str]:
    candidates = sorted(cann_path.glob("*-linux/ascend_toolkit_install.info"))
    candidates.extend(
        path
        for path in (cann_path / "opp" / "version.info", cann_path / "version.info")
        if path.is_file()
    )
    for candidate in candidates:
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?im)^(?:version|Version)\s*=\s*['\"]?([^'\"\s]+)", content)
        if match:
            return match.group(1)
    return None


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cann-path", required=True)
    parser.add_argument(
        "--skip-cann-imports",
        action="store_true",
        help="only validate NumPy (used by the aclnn data scripts)",
    )
    args = parser.parse_args()

    cann_path = pathlib.Path(args.cann_path).resolve()
    cann_version = read_cann_version(cann_path)
    version_label = cann_version or "unknown"
    print(f"[INFO] CANN version: {version_label}", flush=True)

    try:
        numpy = importlib.import_module("numpy")
        importlib.import_module("numpy.core.multiarray")
    except Exception as exc:  # ImportError can be wrapped by NumPy extensions.
        fail(
            "NumPy or numpy.core.multiarray cannot be imported by "
            f"{sys.executable}: {exc}\n"
            "Install the CANN 8.5-compatible build with:\n"
            f"  {sys.executable} -m pip install --upgrade --force-reinstall "
            "'numpy>=1.19.2,<2.0'"
        )

    numpy_version = str(numpy.__version__)
    print(f"[INFO] NumPy: {numpy_version} ({numpy.__file__})", flush=True)
    major_match = re.match(r"(\d+)", numpy_version)
    numpy_major = int(major_match.group(1)) if major_match else None
    if cann_version and cann_version.startswith("8.5") and numpy_major is not None and numpy_major >= 2:
        fail(
            f"CANN {cann_version} requires NumPy <2.0, but {numpy_version} is loaded by "
            f"{sys.executable}. Fix this interpreter with:\n"
            f"  {sys.executable} -m pip install --upgrade --force-reinstall "
            "'numpy>=1.19.2,<2.0'"
        )

    if not args.skip_cann_imports:
        required_modules = (
            "tbe.common.platform",
            "tbe.common.utils",
            "tbe.tikcpp",
            "impl.util.platform_adapter",
        )
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                fail(
                    f"CANN Python module {module_name!r} cannot be imported by "
                    f"{sys.executable}: {exc}\n"
                    f"Ensure {cann_path}/set_env.sh (or bin/setenv.bash) is sourced "
                    "and use the same ASCEND_PYTHON_EXECUTABLE for the whole build."
                )
        print("[INFO] CANN Python modules: OK", flush=True)


if __name__ == "__main__":
    main()
