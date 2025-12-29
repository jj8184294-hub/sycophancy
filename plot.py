import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESPONSIVE_ABS_DELTA = 0.1

def save_delta_hist(df: pd.DataFrame):
    plt.figure()
    plt.hist(df["delta"].to_numpy(), bins=30)
    plt.xlabel("delta = score_pressured - score_neutral")
    plt.ylabel("count")
    plt.title("Pressure-induced agreement shift")
    plt.tight_layout()
    plt.savefig("figs/delta_hist.png", dpi=200)
    plt.close()
def normalize_label_col(df: pd.DataFrame) -> pd.DataFrame:
    s = df["label"].astype(str).str.strip().str.lower()
    s = s.replace({"1": "true", "0": "false", "t": "true", "f": "false"})
    df = df.copy()
    df["label_norm"] = s
    return df
def save_discrimination_plot(df: pd.DataFrame):
    # plot distributions split by label, neutral vs pressured
    plt.figure()
    data = [
        df[df["label"]=="true"]["score_neutral"].to_numpy(),
        df[df["label"]=="false"]["score_neutral"].to_numpy(),
        df[df["label"]=="true"]["score_pressured"].to_numpy(),
        df[df["label"]=="false"]["score_pressured"].to_numpy(),
    ]
    data = [
        df[df["label_norm"]=="true"]["score_neutral"].astype(float).to_numpy(),
        df[df["label_norm"]=="false"]["score_neutral"].astype(float).to_numpy(),
        df[df["label_norm"]=="true"]["score_pressured"].astype(float).to_numpy(),
        df[df["label_norm"]=="false"]["score_pressured"].astype(float).to_numpy(),
    ]
    plt.boxplot(data, labels=["T neutral","F neutral","T pressured","F pressured"])
    plt.ylabel("score = logit(Yes)-logit(No)")
    plt.title("Truth discrimination drops under pressure")
    plt.tight_layout()
    plt.savefig("figs/discrimination.png", dpi=200)
    plt.close()

def save_removal_bar(df: pd.DataFrame):
    resp = df[df["delta"].abs() >= RESPONSIVE_ABS_DELTA].copy()

    # robust mean: clip extreme ratios from near-zero deltas
    def robust_mean(s):
        return float(np.clip(s.to_numpy(), -5, 5).mean())

    means = [
        robust_mean(resp["removed_frac_fix_neuron"]),
        robust_mean(resp["removed_frac_fix_head"]),
        robust_mean(resp["removed_frac_fix_both"]),
    ]
    labels = ["Fix neuron (L21N1876)", "Fix head (L19H2)", "Fix both"]

    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("mean removed fraction (clipped to [-5,5])")
    plt.title(f"Causal removal on responsive subset |delta|≥{RESPONSIVE_ABS_DELTA}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("figs/removal_bar.png", dpi=200)
    plt.close()

def main():
    df = pd.read_csv("results.csv")
    df = normalize_label_col(df)
    save_delta_hist(df)
    save_discrimination_plot(df)
    save_removal_bar(df)

if __name__ == "__main__":
    main()
