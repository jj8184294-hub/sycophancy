#!/usr/bin/env python3
import argparse, os, math
import pandas as pd

BOOL_COLS = [
    "correct_neutral", "correct_pressured", "correct_pressured_edited",
    "pred_neutral", "pred_pressured", "pred_pressured_edited",
]

def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype(bool)
    )

def wilson_ci(k: int, n: int, z: float = 1.96):
    # Wilson score interval for binomial proportion
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z*z)/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="pressure_vector_eval.csv")
    ap.add_argument("--out_dir", default=None, help="optional: save summary csv here")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = to_bool_series(df[c])

    # if multiple alphas exist, we summarize per-alpha; if only one, it's one row anyway
    if "alpha" in df.columns:
        groups = df.groupby("alpha", sort=True)
    else:
        df["alpha"] = float("nan")
        groups = df.groupby("alpha", sort=False)

    rows = []
    for a, g in groups:
        n = len(g)

        kN = int(g["correct_neutral"].sum())
        kP = int(g["correct_pressured"].sum())
        kE = int(g["correct_pressured_edited"].sum())

        accN = kN / n
        accP = kP / n
        accE = kE / n

        ciN = wilson_ci(kN, n)
        ciP = wilson_ci(kP, n)
        ciE = wilson_ci(kE, n)

        rows.append({
            "alpha": a,
            "n": n,
            "acc_neutral": accN,
            "acc_pressured": accP,
            "acc_pressured_edited": accE,
            "acc_drop_vs_neutral": accN - accE,
            "acc_gain_vs_pressured": accE - accP,
            "acc_neutral_ci95_low": ciN[0],
            "acc_neutral_ci95_high": ciN[1],
            "acc_pressured_ci95_low": ciP[0],
            "acc_pressured_ci95_high": ciP[1],
            "acc_edited_ci95_low": ciE[0],
            "acc_edited_ci95_high": ciE[1],
        })

    out = pd.DataFrame(rows).sort_values("alpha")

    # Pretty print
    pd.set_option("display.max_columns", 999)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "accuracy_summary.csv")
        out.to_csv(out_path, index=False)
        print("\nWrote:", out_path)

if __name__ == "__main__":
    main()
