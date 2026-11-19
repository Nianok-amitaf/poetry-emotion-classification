"""Regenerate results/model_comparison.png from results/results.csv."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/results.csv")
x = np.arange(len(df)); w = 0.27
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (col, lab) in enumerate([("cv_accuracy", "5-fold CV"),
                                ("val_accuracy", "Validation"),
                                ("test_accuracy", "Test")]):
    bars = ax.bar(x + (i - 1) * w, df[col], w, label=lab)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=15, ha="right")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.6)
ax.set_title("Poem emotion classification - model comparison (9 classes, n=716)")
ax.axhline(1 / 9, ls="--", c="black", lw=1, label="Random guess (11%)")
ax.legend(); ax.grid(axis="y", ls=":", alpha=.6)
fig.tight_layout(); fig.savefig("results/model_comparison.png", dpi=150)
