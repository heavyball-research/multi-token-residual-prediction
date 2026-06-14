#!/usr/bin/env python3
"""Rename stale parameter prefixes in DeepSpeed intermediate checkpoints.

Some legacy MTP-derived checkpoints saved the MRP module as ``mtp_modules.*``.
The current SDAR-MRP model registers those parameters as ``mrp_modules.*``.
This tool updates DeepSpeed model-state and optimizer-state metadata in place;
Hugging Face ``model*.safetensors`` files are not touched.

The default mode is inspection-only. Pass ``--write`` to atomically replace
affected ``.pt`` files after reviewing the reported changes.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DEFAULT_OLD_PREFIX = "mtp_modules."
DEFAULT_NEW_PREFIX = "mrp_modules."


def _rename_name(name: str, old_prefix: str, new_prefix: str) -> str:
    if name.startswith(old_prefix):
        return new_prefix + name[len(old_prefix):]
    return name


def _rewrite_names(obj: Any, old_prefix: str, new_prefix: str) -> tuple[Any, int]:
    if isinstance(obj, str):
        renamed = _rename_name(obj, old_prefix, new_prefix)
        return renamed, int(renamed != obj)

    if isinstance(obj, Mapping):
        rewritten = obj.__class__()
        count = 0
        for key, value in obj.items():
            new_key, key_count = _rewrite_names(key, old_prefix, new_prefix)
            new_value, value_count = _rewrite_names(value, old_prefix, new_prefix)
            if new_key in rewritten:
                raise ValueError(f"prefix rewrite collides at key {new_key!r}")
            rewritten[new_key] = new_value
            count += key_count + value_count
        return rewritten, count

    if isinstance(obj, list):
        rewritten = []
        count = 0
        for value in obj:
            new_value, value_count = _rewrite_names(value, old_prefix, new_prefix)
            rewritten.append(new_value)
            count += value_count
        return rewritten, count

    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        values = []
        count = 0
        for value in obj:
            new_value, value_count = _rewrite_names(value, old_prefix, new_prefix)
            values.append(new_value)
            count += value_count
        return tuple(values), count

    return obj, 0


def _checkpoint_files(checkpoint: Path) -> list[Path]:
    files = sorted(checkpoint.glob("global_step*/mp_rank_00_model_states.pt"))
    files.extend(sorted(checkpoint.glob("global_step*/*optim_states.pt")))
    if not files:
        raise FileNotFoundError(f"no DeepSpeed state files found under {checkpoint}")
    return files


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".prefix-fix.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def repair_checkpoint(
    checkpoint: Path,
    *,
    old_prefix: str,
    new_prefix: str,
    write: bool,
) -> int:
    changed_files = 0
    for path in _checkpoint_files(checkpoint):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        rewritten, count = _rewrite_names(state, old_prefix, new_prefix)
        if count == 0:
            print(f"unchanged {path}")
            continue
        action = "rewriting" if write else "would rewrite"
        print(f"{action} {path}: {count} renamed name(s)")
        changed_files += 1
        if write:
            _atomic_save(rewritten, path)
    return changed_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX)
    parser.add_argument("--new-prefix", default=DEFAULT_NEW_PREFIX)
    parser.add_argument("--write", action="store_true", help="Replace affected .pt files atomically.")
    args = parser.parse_args()

    changed_files = 0
    for checkpoint in args.checkpoints:
        changed_files += repair_checkpoint(
            checkpoint.resolve(),
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            write=args.write,
        )
    mode = "rewritten" if args.write else "requiring rewrite"
    print(f"{changed_files} DeepSpeed state file(s) {mode}")


if __name__ == "__main__":
    main()
