"""
progress.py — Visualize Flash-MoE experiment progress.
Reads results.tsv, generates progress.png with distinct Q2 and Q4 tracks.

Usage:
    pip install pandas matplotlib
    python progress.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load both results files
    dfs = []
    for path in ["results.tsv", "metal_infer/results.tsv"]:
        if os.path.exists(path):
            try:
                cols = ["commit", "model", "params_B", "active_B", "tok_sec", "ttft_ms", "mem_gb", "status", "description"]
                df = pd.read_csv(path, sep="\t", header=None, names=cols)
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        print("No results.tsv found.")
        sys.exit(0)

    df = pd.concat(dfs, ignore_index=True)
    df["tok_sec"] = pd.to_numeric(df["tok_sec"], errors="coerce")
    df["params_B"] = pd.to_numeric(df["params_B"], errors="coerce")
    df["mem_gb"] = pd.to_numeric(df["mem_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.lower()

    # Filter to 397B model
    is_397b = df["params_B"] >= 300
    df_397b = df[is_397b].copy()

    # Detect Q2 vs Q4 from model name or description
    def get_quant(row):
        model = str(row.get("model", ""))
        desc = str(row.get("description", ""))
        if "2bit" in model.lower() or "2-bit" in desc.lower() or "2bit" in desc.lower():
            return "Q2"
        elif "4bit" in model.lower() or "4-bit" in desc.lower() or "4bit" in desc.lower():
            return "Q4"
        # Heuristic: if tok/s > 5 and kept, likely Q2 era
        if row.get("tok_sec", 0) > 5.0 and row.get("status") == "keep":
            return "Q2"
        return "Q4"  # default to Q4

    df_397b["quant"] = df_397b.apply(get_quant, axis=1)

    n_total = len(df)
    n_397b = len(df_397b)
    n_q2 = len(df_397b[df_397b["quant"] == "Q2"])
    n_q4 = len(df_397b[df_397b["quant"] == "Q4"])
    kept = df_397b[df_397b["status"] == "keep"]

    print(f"\n=== Flash-MoE: 397B Model Journey ===")
    print(f"Total experiments: {n_total} ({n_397b} on 397B: {n_q2} Q2, {n_q4} Q4)")

    for q in ["Q2", "Q4"]:
        qk = kept[kept["quant"] == q]
        if len(qk) > 0:
            best = qk.loc[qk["tok_sec"].idxmax()]
            print(f"Best {q}: {best['tok_sec']:.2f} tok/s — {best['description'][:70]}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(16, 8))

    # Color scheme: Q2 = blue family, Q4 = orange/green family
    colors = {
        ("Q2", "keep"):    "#2196F3",   # blue
        ("Q2", "discard"): "#90CAF9",   # light blue
        ("Q4", "keep"):    "#FF9800",   # orange
        ("Q4", "discard"): "#FFCC80",   # light orange
    }

    # Assign x-positions (experiment index within each track)
    for quant in ["Q4", "Q2"]:
        qdf = df_397b[df_397b["quant"] == quant].copy()
        if len(qdf) == 0:
            continue

        # Assign sequential x positions
        x_pos = list(range(len(qdf)))

        for status in ["discard", "keep"]:
            mask = (qdf["status"] == status) & (qdf["tok_sec"] > 0)
            if not mask.any():
                continue
            subset = qdf[mask]
            xs = [x_pos[i] for i, idx in enumerate(qdf.index) if idx in subset.index]
            color = colors.get((quant, status), "#999")
            is_keep = status == "keep"
            ax.scatter(xs, subset["tok_sec"],
                       c=color,
                       s=100 if is_keep else 30,
                       label=f"{quant} {status}",
                       zorder=5 if is_keep else 3,
                       edgecolors="black" if is_keep else "none",
                       linewidths=0.7 if is_keep else 0,
                       alpha=0.95 if is_keep else 0.5,
                       marker="o" if quant == "Q4" else "D")

        # Running best line for kept experiments
        kept_q = qdf[(qdf["status"] == "keep") & (qdf["tok_sec"] > 0)].copy()
        if len(kept_q) > 1:
            running_best = kept_q["tok_sec"].cummax()
            xs_best = []
            for i, idx in enumerate(qdf.index):
                if idx in kept_q.index:
                    xs_best.append(x_pos[i])
            if len(xs_best) == len(running_best):
                line_color = "#1565C0" if quant == "Q2" else "#E65100"
                ax.step(xs_best, running_best.values,
                        where="post", color=line_color, linewidth=2.5, alpha=0.8,
                        label=f"Running best ({quant})")

    # Best markers
    for quant, color, yoff in [("Q2", "#1565C0", 0.3), ("Q4", "#E65100", -0.3)]:
        qk = kept[kept["quant"] == quant]
        if len(qk) > 0:
            best = qk.loc[qk["tok_sec"].idxmax()]
            ax.axhline(y=best["tok_sec"], color=color, linestyle="--", alpha=0.3, linewidth=1)
            ax.text(0.98, best["tok_sec"] + yoff,
                    f'  {quant} best: {best["tok_sec"]:.2f} tok/s',
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=10, color=color, alpha=0.8,
                    fontweight="bold")

    ax.set_ylabel("Tokens/second", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_title("Flash-MoE: Running Qwen3.5-397B on a MacBook Pro (M3 Max, 48GB)\n"
                 f"Q2: 2-bit experts (speed) | Q4: 4-bit experts (quality) | {n_397b} experiments",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=-0.5, top=max(df_397b["tok_sec"].max() + 1, 8))

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved progress.png")


if __name__ == "__main__":
    main()
