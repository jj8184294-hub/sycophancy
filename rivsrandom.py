#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_csv", required=True)
    ap.add_argument("--ridge_csv", required=True)
    ap.add_argument("--out_prefix", default="ridge_vs_random")
    ap.add_argument("--alpha", type=float, default=None, help="Filter to a single alpha (recommended).")
    args = ap.parse_args()

    rand = pd.read_csv(args.random_csv)
    ridge = pd.read_csv(args.ridge_csv)

    if args.alpha is not None:
        rand = rand[np.isclose(rand["alpha"].astype(float), args.alpha)]
        ridge = ridge[np.isclose(ridge["alpha"].astype(float), args.alpha)]

    assert len(ridge) == 1, f"ridge_csv must contain exactly 1 row after filtering; got {len(ridge)}"

    r_mean = float(ridge["mean_delta_reduction"].iloc[0])
    r_frac = float(ridge["frac_improved"].iloc[0])

    x = rand["mean_delta_reduction"].astype(float).to_numpy()
    x_mean = float(np.mean(x))
    x_std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    x_max = float(np.max(x))
    z = (r_mean - x_mean) / x_std if x_std > 0 else float("inf")

    # empirical one-sided p-value: P(random >= ridge)
    p_emp = (1.0 + float(np.sum(x >= r_mean))) / (len(x) + 1.0)

    print(f"[alpha={float(ridge['alpha'].iloc[0]):g}] ridge_mean={r_mean:.6f}")
    print(f"random_mean={x_mean:.6f} random_std={x_std:.6f} random_max={x_max:.6f}")
    print(f"z_score_vs_random={z:.2f}")
    print(f"empirical_one_sided_p<= {p_emp:.6f}  (based on {len(x)} random draws)")

    # 1) Histogram of random mean_delta_reduction + ridge line
    plt.figure()
    plt.hist(x, bins=20, alpha=0.75, label=f"Random directions (n={len(x)})")
    plt.axvline(
        r_mean,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"Ridge direction (mean_delta_reduction={r_mean:.3f})",
    )
    plt.title("mean_delta_reduction: random directions vs ridge")
    plt.xlabel("mean_delta_reduction (delta - delta_edited)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_hist_mean_delta_reduction.png", dpi=200)


    # 2) Dot/rank plot
    plt.figure()
    xs = np.arange(len(x))
    plt.scatter(xs, np.sort(x), s=15)
    plt.axhline(r_mean)
    plt.title("Rank plot: random draws (sorted) + ridge (horizontal line)")
    plt.xlabel("random draw rank")
    plt.ylabel("mean_delta_reduction")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_rank_mean_delta_reduction.png", dpi=200)

    # 3) frac_improved distribution + ridge
    if "frac_improved" in rand.columns:
        f = rand["frac_improved"].astype(float).to_numpy()

        plt.figure()
        plt.hist(f, bins=20, range=(0, 1))
        plt.axvline(r_frac)
        plt.title("frac_improved: random directions vs ridge")
        plt.xlabel("frac_improved")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_hist_frac_improved.png", dpi=200)

        plt.figure()
        plt.scatter(np.arange(len(f)), np.sort(f), s=15)
        plt.axhline(r_frac)
        plt.title("Rank plot: frac_improved random draws + ridge")
        plt.xlabel("random draw rank")
        plt.ylabel("frac_improved")
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_rank_frac_improved.png", dpi=200)

if __name__ == "__main__":
    main()
