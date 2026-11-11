import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (tag used by train.py, label, color) - a run keeps its color in every chart
RUNS = [
    ("basic", "11 inputs", "#eb6834"),
    ("space", "14 inputs (+ free space)", "#1baf7a"),
]
GRAY = "#8a8a8a"


def moving_average(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


random_scores = np.loadtxt("results/random_scores.txt")

# 1. score per game while training
fig, ax = plt.subplots(figsize=(8, 5))
for tag, label, color in RUNS:
    path = f"results/{tag}_train_scores.txt"
    if not os.path.exists(path):
        continue
    scores = np.loadtxt(path)
    ma = moving_average(scores)
    ax.plot(scores, color=color, alpha=0.15, linewidth=0.8)
    ax.plot(np.arange(len(ma)) + 49, ma, color=color, linewidth=2, label=f"{label} (last 50 games: {scores[-50:].mean():.1f})")
ax.axhline(random_scores.mean(), color=GRAY, linestyle="--", linewidth=1.5, label=f"random moves ({random_scores.mean():.2f})")
ax.set_xlabel("game")
ax.set_ylabel("score (food eaten)")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("results/scores.png", dpi=150)

# 2. final test: 100 games without random moves
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(0, 66, 3)
random_100 = random_scores[:100]  # same number of games as the trained agents
ax.hist(random_100, bins=bins, color=GRAY, alpha=0.6, label=f"random moves (mean {random_100.mean():.1f})")
for tag, label, color in RUNS:
    path = f"results/{tag}_test_scores.txt"
    if not os.path.exists(path):
        continue
    test = np.loadtxt(path)
    ax.hist(test, bins=bins, color=color, alpha=0.6, label=f"{label} (mean {test.mean():.1f}, best {int(test.max())})")
ax.set_xlabel("score in one game")
ax.set_ylabel("number of games")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("results/test_scores.png", dpi=150)