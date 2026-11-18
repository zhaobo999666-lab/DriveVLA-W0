#!/usr/bin/env python3
"""Render a traceable ZeRO-2 config for performance A/B runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_bucket(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        bucket = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bucket must be 'auto' or a positive integer") from exc
    if bucket <= 0:
        raise argparse.ArgumentTypeError("bucket must be positive")
    return bucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reduce-bucket-size", default="auto", type=parse_bucket)
    parser.add_argument("--overlap-comm", choices=("0", "1"), default="1")
    args = parser.parse_args()

    with args.source.open(encoding="utf-8") as stream:
        config = json.load(stream)
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") != 2:
        raise SystemExit(f"expected a ZeRO-2 source config: {args.source}")

    zero["reduce_bucket_size"] = args.reduce_bucket_size
    zero["overlap_comm"] = args.overlap_comm == "1"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
