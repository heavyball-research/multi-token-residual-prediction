#!/usr/bin/env python3
"""Overlay per-layer input->output cosine similarity for several SDAR target models.

Reads JSON outputs of test_profile_layer_io_similarity.py (one per model) and
draws a single figure. With --normalize-depth, x is depth fraction (0..1) so
models with different layer counts overlay.

Usage:
  python scripts/plot_layer_io_similarity.py \
      --json 1.7B=cc_logs/layer_io_cossim_1_7b.json \
      --json 4B=cc_logs/layer_io_cossim_4b.json \
      --json 8B=cc_logs/layer_io_cossim_8b.json \
      --out cc_logs/layer_io_cossim.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="append", required=True,
                   help="LABEL=path.json (repeatable).")
    p.add_argument("--out", default="cc_logs/layer_io_cossim.png")
    p.add_argument("--normalize-depth", action="store_true")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for spec in args.json:
        label, path = spec.split("=", 1)
        rows = sorted(json.load(open(path)), key=lambda r: r["layer"])
        ys = [r["cos_in_out"] for r in rows]
        n = len(ys)
        xs = [i / (n - 1) for i in range(n)] if args.normalize_depth else [r["layer"] for r in rows]
        ax.plot(xs, ys, marker="o", ms=8, lw=3.0, label=f"{label} ({n} layers)")

    ax.set_xlabel("depth fraction" if args.normalize_depth else "layer index", fontsize=18)
    ax.set_ylabel("cos(input, output) activation similarity", fontsize=18)
    ax.set_title("SDAR target model: per-layer input→output activation similarity",
                 fontsize=19)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
