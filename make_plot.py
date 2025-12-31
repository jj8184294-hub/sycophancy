#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def responsive_mask(df: pd.DataFrame, thr: float = 0.1):
    d = df["delta"].astype(float)
    return d.abs() >= thr

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--responsive_thr", type=float, default=0.1)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    # Normalize labels
    df["label_norm"] = df["label"].astype(str).str.lower().map(lambda x: "true" if x == "true" else ("false" if x == "false" else x))

    # -----------------------
    # Fig 1: Δ histogram
    # -----------------------
    deltas = df["delta"].astype(float).dropna().values
    plt.figure()
    plt.hist(deltas, bins=30)
    plt.xlabel("Δ = score_pressured - score_neutral")
    plt.ylabel("count")
    plt.title("Pressure effect distribution (Δ)")
    savefig(os.path.join(args.out_dir, "fig1_delta_hist.png"))

    # -----------------------
    # Fig 2: score boxplot
    # -----------------------
    sn = df["score_neutral"].astype(float).dropna().values
    sp = df["score_pressured"].astype(float).dropna().values
    plt.figure()
    plt.boxplot([sn, sp], labels=["neutral", "pressured"], showfliers=False)
    plt.ylabel('score = logit(" yes") - logit(" no")')
    plt.title("Yes/No score under neutral vs pressured prompts")
    savefig(os.path.join(args.out_dir, "fig2_score_boxplot.png"))

    # -----------------------
    # Fig 3: truth discrimination bars
    # -----------------------
    def mean_for(label, col):
        x = df.loc[df["label_norm"] == label, col].astype(float).dropna()
        return float(x.mean()) if len(x) else np.nan

    m_true_n = mean_for("true", "score_neutral")
    m_false_n = mean_for("false", "score_neutral")
    m_true_p = mean_for("true", "score_pressured")
    m_false_p = mean_for("false", "score_pressured")

    # 4 bars: (true/false) x (neutral/pressured)
    plt.figure()
    vals = [m_true_n, m_false_n, m_true_p, m_false_p]
    labels = ["true (neutral)", "false (neutral)", "true (pressured)", "false (pressured)"]
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=20, ha="right")
    plt.ylabel('mean score = logit(" yes") - logit(" no")')
    plt.title("Truth separation under neutral vs pressured")
    savefig(os.path.join(args.out_dir, "fig3_truth_bars.png"))

    # -----------------------
    # Fig 4: recovery distributions (responsive subset)
    # -----------------------
    mask = responsive_mask(df, args.responsive_thr)

    for col, name in [
        ("best_neuron_recovery", "neuron"),
        ("best_head_recovery", "head"),
        ("best_layer_recovery", "layer"),
    ]:
        x = df.loc[mask, col].astype(float).dropna().values
        if len(x) == 0:
            continue

        # CDF plot (more informative than mean when heavy-tailed)
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(f"{col} (responsive |Δ| >= {args.responsive_thr})")
        plt.ylabel("CDF")
        plt.title(f"CDF of recovery: {name}")
        savefig(os.path.join(args.out_dir, f"fig4_cdf_recovery_{name}.png"))

    # -----------------------
    # Fig 5: counts (best layer + top neuron IDs)
    # -----------------------
    # Best layer counts
    plt.figure()
    bl = df["best_layer"].dropna().astype(int)
    vc = bl.value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values)
    plt.xlabel("best_layer")
    plt.ylabel("count")
    plt.title("Which layer best recovers the pressure effect?")
    savefig(os.path.join(args.out_dir, "fig5a_best_layer_counts.png"))

    # Top neuron identities (layer, neuron)
    plt.figure()
    nn = df[["best_neuron_layer", "best_neuron"]].dropna()
    nn["best_neuron_layer"] = nn["best_neuron_layer"].astype(int)
    nn["best_neuron"] = nn["best_neuron"].astype(int)
    key = nn.apply(lambda r: f"L{r['best_neuron_layer']}N{r['best_neuron']}", axis=1)
    top = key.value_counts().head(10)
    plt.bar(np.arange(len(top)), top.values)
    plt.xticks(np.arange(len(top)), top.index, rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Top 10 most-selected 'best neuron' identities")
    savefig(os.path.join(args.out_dir, "fig5b_top_neuron_counts.png"))

    # Top head identities (layer, head)
    plt.figure()
    hh = df[["best_head_layer", "best_head"]].dropna()
    hh["best_head_layer"] = hh["best_head_layer"].astype(int)
    hh["best_head"] = hh["best_head"].astype(int)
    hkey = hh.apply(lambda r: f"L{r['best_head_layer']}H{r['best_head']}", axis=1)
    htop = hkey.value_counts().head(10)
    plt.bar(np.arange(len(htop)), htop.values)
    plt.xticks(np.arange(len(htop)), htop.index, rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Top 10 most-selected 'best head' identities")
    savefig(os.path.join(args.out_dir, "fig5c_top_head_counts.png"))

    # -----------------------
    # Quick textual summary saved to disk
    # -----------------------
    out_txt = os.path.join(args.out_dir, "summary_n1.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"n_total = {len(df)}\n")
        f.write(f"mean_delta = {float(np.mean(deltas)):.4f}\n")
        f.write(f"median_delta = {float(np.median(deltas)):.4f}\n")
        f.write(f"frac_delta_positive = {float(np.mean(deltas>0)):.4f}\n")
        f.write(f"n_responsive(|Δ|>={args.responsive_thr}) = {int(mask.sum())}\n")
        f.write("\nTop neurons:\n")
        f.write(top.to_string())
        f.write("\n\nTop heads:\n")
        f.write(htop.to_string())
        f.write("\n")

    print("Wrote figures + summary to:", args.out_dir)

if __name__ == "__main__":
    main()
