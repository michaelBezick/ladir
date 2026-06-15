#!/usr/bin/env python
"""Fail early when the active venv cannot import core training dependencies."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import typing_extensions
        from typing_extensions import TypeIs  # noqa: F401
    except Exception as exc:
        location = getattr(sys.modules.get("typing_extensions"), "__file__", "not importable")
        print(
            "Invalid Python environment: torch 2.8.0 requires a newer "
            "typing_extensions package with TypeIs.\n"
            f"Resolved typing_extensions from: {location}\n"
            "Fix the venv used by this job, for example:\n"
            "  python -m pip install --upgrade -r requirements.txt\n"
            "or:\n"
            "  python -m pip install --upgrade typing_extensions==4.15.0",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        import torch
    except Exception as exc:
        print(
            "Invalid Python environment: failed to import torch after validating "
            "typing_extensions.\n"
            f"Python executable: {sys.executable}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(
        "Python environment OK: "
        f"python={sys.executable}, "
        f"typing_extensions={typing_extensions.__file__}, "
        f"torch={torch.__version__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
