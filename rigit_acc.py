#!/usr/bin/env python3
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

BOOL_COLS = [
    "pred_neutral", "pred_pressured", "pred_pressured_edited",
    "correct_neutral", "correct_pressured", "correct_pressured_edited",
]

def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    # handles "True"/"False", "true"/"false", 1/0
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).astype(bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="pressure_vector_eval.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title_prefix", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # normalize bool columns (in case they are strings)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = to_bool_series(df[c])

    if "alpha" not in df.columns:
        raise ValueError("CSV must contain an 'alpha' column.")

    # accuracy by alpha
    g = df.groupby("alpha").agg(
        acc_neutral=("correct_neutral", "mean"),
        acc_pressured=("correct_pressured", "mean"),
        acc_edited=("correct_pressured_edited", "mean"),
        n=("correct_neutral", "size"),
    ).reset_index().sort_values("alpha")

    # derived deltas (these are what you described in your summary CSV)
    g["acc_drop_vs_neutral"] = g["acc_neutral"] - g["acc_edited"]
    g["acc_gain_vs_pressured"] = g["acc_edited"] - g["acc_pressured"]

    # ---- Plot 1: accuracy curves ----
    plt.figure()
    plt.plot(g["alpha"], g["acc_neutral"], marker="o", label="Neutral acc")
    plt.plot(g["alpha"], g["acc_pressured"], marker="o", label="Pressured acc")
    plt.plot(g["alpha"], g["acc_edited"], marker="o", label="Pressured+Edited acc")
    plt.ylim(0, 1)
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    title = "Accuracy vs alpha (neutral / pressured / pressured+edited)"
    if args.title_prefix:
        title = f"{args.title_prefix} — {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "accuracy_vs_alpha.png")
    plt.savefig(out1, dpi=200)

    # ---- Plot 2: accuracy change (edited relative to baselines) ----
    plt.figure()
    plt.plot(g["alpha"], g["acc_drop_vs_neutral"], marker="o", label="(neutral - edited)  [drop vs neutral]")
    plt.plot(g["alpha"], g["acc_gain_vs_pressured"], marker="o", label="(edited - pressured) [gain vs pressured]")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("accuracy delta")
    title = "Accuracy change caused by edit"
    if args.title_prefix:
        title = f"{args.title_prefix} — {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "accuracy_change_vs_alpha.png")
    plt.savefig(out2, dpi=200)

    # save summary table
    out_csv = os.path.join(args.out_dir, "accuracy_summary_by_alpha.csv")
    g.to_csv(out_csv, index=False)

    # print a compact summary (useful for README/paper)
    print("Wrote:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out_csv)
    print("\nSummary:")
    print(g.to_string(index=False))

if __name__ == "__main__":
    main()
