#!/usr/bin/env python3
"""
make_figs_fixed.py

Plots for pressure-localization runs, using *fixed* candidate recoveries
(to avoid "best_*" selection bias).

Expected columns in per_claim.csv:
  - score_neutral, score_pressured, delta, label
  - best_layer, best_neuron_layer, best_neuron, best_head_layer, best_head (optional; for count plots)
  - fixed_layer_resid_recovery
  - fixed_head_recovery
  - fixed_neuron_recovery

Usage:
  python make_figs_fixed.py --csv out/per_claim.csv --out_dir figs --responsive_thr 0.1
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def responsive_mask(df: pd.DataFrame, thr: float = 0.1):
    d = pd.to_numeric(df["delta"], errors="coerce")
    return d.abs() >= thr


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def cdf_xy(x: np.ndarray):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--responsive_thr", type=float, default=0.1)
    ap.add_argument("--clip_recovery", type=float, default=5.0, help="Clip recovery to [-clip, clip] for plots.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    # Normalize labels
    df["label_norm"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .map(lambda x: "true" if x == "true" else ("false" if x == "false" else x))
    )

    # -----------------------
    # Fig 1: Δ histogram
    # -----------------------
    deltas = _to_num(df["delta"]).dropna().values
    plt.figure()
    plt.hist(deltas, bins=30)
    plt.xlabel("Δ = score_pressured - score_neutral")
    plt.ylabel("count")
    plt.title("Pressure effect distribution (Δ)")
    savefig(os.path.join(args.out_dir, "fig1_delta_hist.png"))

    # -----------------------
    # Fig 2: score boxplot
    # -----------------------
    sn = _to_num(df["score_neutral"]).dropna().values
    sp = _to_num(df["score_pressured"]).dropna().values
    plt.figure()
    plt.boxplot([sn, sp], labels=["neutral", "pressured"], showfliers=False)
    plt.ylabel('score = logit(" yes") - logit(" no")')
    plt.title("Yes/No score under neutral vs pressured prompts")
    savefig(os.path.join(args.out_dir, "fig2_score_boxplot.png"))

    # -----------------------
    # Fig 3: mean scores by (truth label x condition)
    # (This is the data that underlies D = mean(true) - mean(false).)
    # -----------------------
    def mean_for(label, col):
        x = _to_num(df.loc[df["label_norm"] == label, col]).dropna()
        return float(x.mean()) if len(x) else np.nan

    m_true_n = mean_for("true", "score_neutral")
    m_false_n = mean_for("false", "score_neutral")
    m_true_p = mean_for("true", "score_pressured")
    m_false_p = mean_for("false", "score_pressured")

    plt.figure()
    vals = [m_true_n, m_false_n, m_true_p, m_false_p]
    labels = ["true (neutral)", "false (neutral)", "true (pressured)", "false (pressured)"]
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=20, ha="right")
    plt.ylabel('mean score = logit(" yes") - logit(" no")')
    plt.title("Mean Yes/No score by label under neutral vs pressured")
    savefig(os.path.join(args.out_dir, "fig3_truth_bars.png"))

    # -----------------------
    # Fig 4: FIXED candidate recovery distributions (responsive subset)
    # -----------------------
    mask = responsive_mask(df, args.responsive_thr)

    fixed_cols = [
        ("fixed_layer_resid_recovery", "fixed layer resid_post (e.g., L19)"),
        ("fixed_head_recovery", "fixed attention head (e.g., L19H2)"),
        ("fixed_neuron_recovery", "fixed MLP neuron (e.g., L21N1876)"),
    ]

    # 4a: CDFs (one plot per fixed candidate)
    for col, title in fixed_cols:
        if col not in df.columns:
            continue
        x = _to_num(df.loc[mask, col]).dropna().values
        if x.size == 0:
            continue
        if args.clip_recovery and args.clip_recovery > 0:
            x = np.clip(x, -args.clip_recovery, args.clip_recovery)

        xs, ys = cdf_xy(x)
        if xs is None:
            continue

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(f"{col} (responsive |Δ| >= {args.responsive_thr})")
        plt.ylabel("CDF (fraction of claims ≤ x)")
        plt.title(f"CDF of recovery (fixed candidate): {title}")
        savefig(os.path.join(args.out_dir, f"fig4_cdf_{col}.png"))

    # 4b: Overlaid histograms for fixed recoveries (helps intuition)
    # (All on same axes; clipped to keep tails from wrecking the view.)
    xs_list = []
    names = []
    for col, title in fixed_cols:
        if col not in df.columns:
            continue
        x = _to_num(df.loc[mask, col]).dropna().values
        if x.size == 0:
            continue
        if args.clip_recovery and args.clip_recovery > 0:
            x = np.clip(x, -args.clip_recovery, args.clip_recovery)
        xs_list.append(x)
        names.append(col)

    if xs_list:
        plt.figure()
        plt.hist(xs_list, bins=40, label=names, histtype="step")
        plt.xlabel(f"recovery (clipped to [-{args.clip_recovery}, {args.clip_recovery}])")
        plt.ylabel("count")
        plt.title("Fixed candidate recovery distributions (responsive subset)")
        plt.legend()
        savefig(os.path.join(args.out_dir, "fig4b_fixed_recovery_hists.png"))

    # 4c: Fraction of claims where fixed candidate reaches certain recovery thresholds
    # (Quick “how often does it recover >= 0.5 / 0.9 / 1.0” table.)
    thresholds = [0.25, 0.5, 0.9, 1.0]
    rows = []
    for col, _title in fixed_cols:
        if col not in df.columns:
            continue
        x = _to_num(df.loc[mask, col]).dropna().values
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        row = {"col": col, "n": int(x.size)}
        for t in thresholds:
            row[f"frac_ge_{t}"] = float(np.mean(x >= t))
        rows.append(row)

    frac_path = os.path.join(args.out_dir, "fig4c_fixed_recovery_thresholds.csv")
    if rows:
        pd.DataFrame(rows).to_csv(frac_path, index=False)

    # -----------------------
    # Fig 5: optional count plots (best-of diagnostics)
    # -----------------------
    # Best layer counts
    if "best_layer" in df.columns:
        bl = _to_num(df["best_layer"]).dropna().astype(int)
        if len(bl):
            plt.figure()
            vc = bl.value_counts().sort_index()
            plt.bar(vc.index.astype(str), vc.values)
            plt.xlabel("best_layer")
            plt.ylabel("count")
            plt.title("Which layer most often 'wins' as best_layer?")
            savefig(os.path.join(args.out_dir, "fig5a_best_layer_counts.png"))

    # Top neuron identities (layer, neuron)
    if "best_neuron_layer" in df.columns and "best_neuron" in df.columns:
        nn = df[["best_neuron_layer", "best_neuron"]].copy()
        nn["best_neuron_layer"] = _to_num(nn["best_neuron_layer"])
        nn["best_neuron"] = _to_num(nn["best_neuron"])
        nn = nn.dropna()
        if len(nn):
            nn["best_neuron_layer"] = nn["best_neuron_layer"].astype(int)
            nn["best_neuron"] = nn["best_neuron"].astype(int)
            key = nn.apply(lambda r: f"L{r['best_neuron_layer']}N{r['best_neuron']}", axis=1)
            top = key.value_counts().head(10)

            plt.figure()
            plt.bar(np.arange(len(top)), top.values)
            plt.xticks(np.arange(len(top)), top.index, rotation=30, ha="right")
            plt.ylabel("count")
            plt.title("Top 10 most-selected 'best neuron' identities")
            savefig(os.path.join(args.out_dir, "fig5b_top_neuron_counts.png"))

    # Top head identities (layer, head)
    if "best_head_layer" in df.columns and "best_head" in df.columns:
        hh = df[["best_head_layer", "best_head"]].copy()
        hh["best_head_layer"] = _to_num(hh["best_head_layer"])
        hh["best_head"] = _to_num(hh["best_head"])
        hh = hh.dropna()
        if len(hh):
            hh["best_head_layer"] = hh["best_head_layer"].astype(int)
            hh["best_head"] = hh["best_head"].astype(int)
            hkey = hh.apply(lambda r: f"L{r['best_head_layer']}H{r['best_head']}", axis=1)
            htop = hkey.value_counts().head(10)

            plt.figure()
            plt.bar(np.arange(len(htop)), htop.values)
            plt.xticks(np.arange(len(htop)), htop.index, rotation=30, ha="right")
            plt.ylabel("count")
            plt.title("Top 10 most-selected 'best head' identities")
            savefig(os.path.join(args.out_dir, "fig5c_top_head_counts.png"))

    # -----------------------
    # Quick textual summary saved to disk
    # -----------------------
    out_txt = os.path.join(args.out_dir, "summary_plots.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"n_total = {len(df)}\n")
        f.write(f"mean_delta = {float(np.mean(deltas)):.4f}\n")
        f.write(f"median_delta = {float(np.median(deltas)):.4f}\n")
        f.write(f"frac_delta_positive = {float(np.mean(deltas > 0)):.4f}\n")
        f.write(f"n_responsive(|Δ| >= {args.responsive_thr}) = {int(mask.sum())}\n")
        f.write(f"clip_recovery = {args.clip_recovery}\n\n")

        for col, _title in fixed_cols:
            if col not in df.columns:
                continue
            x = _to_num(df.loc[mask, col]).dropna().values
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            f.write(f"{col}: n={x.size} mean={float(np.mean(x)):.4f} median={float(np.median(x)):.4f}\n")

        f.write("\nWrote recovery threshold table to:\n")
        f.write(frac_path + "\n" if rows else "(no fixed recovery threshold table written)\n")

    print("Wrote figures + summaries to:", args.out_dir)
    if rows:
        print("Also wrote threshold summary CSV:", frac_path)


if __name__ == "__main__":
    main()
