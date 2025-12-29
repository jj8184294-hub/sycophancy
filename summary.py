import pandas as pd
import numpy as np

RESPONSIVE_ABS_DELTA = 0.1

# def discrimination(df: pd.DataFrame, col: str) -> float:
#     t = df[df["label"] == "true"][col].mean()
#     f = df[df["label"] == "false"][col].mean()
#     return float(t - f)
def normalize_label_col(df: pd.DataFrame) -> pd.DataFrame:
    # Works if labels are bool, "true"/"false", "True"/"False", 1/0, etc.
    s = df["label"].astype(str).str.strip().str.lower()
    # Map common variants
    s = s.replace({"1": "true", "0": "false", "t": "true", "f": "false"})
    df = df.copy()
    df["label_norm"] = s
    return df

def discrimination(df: pd.DataFrame, col: str) -> float:
    t = df.loc[df["label_norm"] == "true", col].astype(float).mean()
    f = df.loc[df["label_norm"] == "false", col].astype(float).mean()
    return float(t - f)


def main():
    # df = pd.read_csv("results.csv")
    df = pd.read_csv("results.csv")
    df = normalize_label_col(df)

    # Basic pressure effect
    mean_delta = df["delta"].mean()
    median_delta = df["delta"].median()
    frac_pos = (df["delta"] > 0).mean()

    # Truth discrimination under neutral vs pressured
    D_neutral = discrimination(df, "score_neutral")
    D_pressured = discrimination(df, "score_pressured")

    # Responsive subset for removed_frac stability
    resp = df[df["delta"].abs() >= RESPONSIVE_ABS_DELTA].copy()

    # Optionally clip removed fractions to reduce outlier dominance
    def clipped_mean(s, lo=-5, hi=5):
        return float(np.clip(s.to_numpy(), lo, hi).mean())

    out = {
        "n_total": len(df),
        "mean_delta": float(mean_delta),
        "median_delta": float(median_delta),
        "frac_delta_positive": float(frac_pos),
        "D_neutral": float(D_neutral),
        "D_pressured": float(D_pressured),
        "D_drop": float(D_pressured - D_neutral),
        "n_responsive": len(resp),
        "mean_removed_frac_neuron": float(resp["removed_frac_fix_neuron"].mean()),
        "mean_removed_frac_head": float(resp["removed_frac_fix_head"].mean()),
        "mean_removed_frac_both": float(resp["removed_frac_fix_both"].mean()),
        "clipped_mean_removed_frac_neuron": clipped_mean(resp["removed_frac_fix_neuron"]),
        "clipped_mean_removed_frac_head": clipped_mean(resp["removed_frac_fix_head"]),
        "clipped_mean_removed_frac_both": clipped_mean(resp["removed_frac_fix_both"]),
    }

    print("=== Summary ===")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
