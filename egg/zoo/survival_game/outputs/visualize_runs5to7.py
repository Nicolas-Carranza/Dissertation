#!/usr/bin/env python3
"""
Visualize training results from runs 5, 6, and 7 — comparative analysis.
Generates publication-quality figures for supervisor presentation.

Run 5: 40-class recon fix (baseline for action policy experiments)
Run 6: + action entropy (0.1), action temperature (2.0), reward normalisation
Run 7: + higher entropy (0.5), lower action temp (1.0), no temp decay
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "outputs"
FIG_DIR = BASE_DIR / "figures_runs5to7"
FIG_DIR.mkdir(exist_ok=True)

RUN_FILES = {
    "Run 5": BASE_DIR / "run5.txt",
    "Run 6": BASE_DIR / "run6.txt",
    "Run 7": BASE_DIR / "run7.txt",
}

# Run descriptions for legends
RUN_LABELS = {
    "Run 5": "Run 5 — 40-class recon (baseline)",
    "Run 6": "Run 6 — +entropy 0.1, temp 2.0, norm",
    "Run 7": "Run 7 — +entropy 0.5, temp 1.0, no decay",
}

# Colour palette
RUN_COLORS = {
    "Run 5": "#2c3e50",  # dark blue-grey
    "Run 6": "#e67e22",  # orange
    "Run 7": "#27ae60",  # green
}

RUN_MARKERS = {
    "Run 5": "o",
    "Run 6": "s",
    "Run 7": "D",
}

# Baselines from prototype.py
BASELINE_RANDOM = 21.1
BASELINE_GREEDY = 55.5
BASELINE_OPTIMAL = 63.1

# Action colour palette (colour-blind friendly)
ACTION_COLORS = {
    "hunt": "#e41a1c",
    "gather": "#4daf4a",
    "flee": "#377eb8",
    "rest": "#984ea3",
    "eat": "#ff7f00",
    "craft": "#a65628",
    "mitigate": "#999999",
    "endure": "#f781bf",
}


# ── Parse a single log file ───────────────────────────────────────
def parse_run(filepath):
    """Parse a run output file and return train_logs, test_logs, eval_blocks, msg_analyses."""
    train_logs = []
    test_logs = []
    eval_blocks = []
    msg_analyses = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # JSON log lines
        if line.startswith("{"):
            try:
                d = json.loads(line)
                if d.get("mode") == "train":
                    train_logs.append(d)
                elif d.get("mode") == "test":
                    test_logs.append(d)
            except json.JSONDecodeError:
                pass

        # Evaluation blocks
        if "Survival Game Evaluation" in line:
            block = {}
            while i < len(lines):
                l = lines[i].strip()
                m = re.search(r"Epoch\s+(\d+)", l)
                if m:
                    block["epoch"] = int(m.group(1))
                m = re.search(r"Survival rate:\s+([\d.]+)%", l)
                if m:
                    block["survival"] = float(m.group(1))
                m = re.search(r"Mean reward:\s+([\d.]+)", l)
                if m:
                    block["mean_reward"] = float(m.group(1))
                # Action distribution
                for action in ["hunt", "gather", "flee", "rest", "mitigate",
                               "endure", "eat", "craft_spear", "craft_fire",
                               "craft_shelter", "craft_rod"]:
                    m = re.search(rf"^\s*{action}:\s+([\d.]+)%", l)
                    if m:
                        block[f"eval_{action}"] = float(m.group(1))
                if l.startswith("=" * 10) and "epoch" in block and "survival" in block:
                    eval_blocks.append(block)
                    break
                i += 1

        # Message Analysis blocks
        if "Message Analysis" in line:
            m_epoch = re.search(r"Epoch\s+(\d+)", line)
            analysis = {"epoch": int(m_epoch.group(1)) if m_epoch else 0, "entities": {}}
            i += 1  # skip header line
            i += 1  # skip separator line
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("=") or l.startswith("{"):
                    break
                # Parse: EntityType | Count | Unique | Top messages
                m = re.match(r"(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", l)
                if m:
                    analysis["entities"][m.group(1)] = {
                        "count": int(m.group(2)),
                        "unique": int(m.group(3)),
                    }
                i += 1
            if analysis["entities"]:
                msg_analyses.append(analysis)
            continue  # don't increment i again

        i += 1

    return train_logs, test_logs, eval_blocks, msg_analyses


# ── Parse all runs ─────────────────────────────────────────────────
all_data = {}
for run_name, filepath in RUN_FILES.items():
    train, test, evals, msgs = parse_run(filepath)
    all_data[run_name] = {
        "train": train, "test": test, "evals": evals, "msgs": msgs
    }
    print(f"Parsed {run_name}: {len(train)} train, {len(test)} test, "
          f"{len(evals)} evals, {len(msgs)} msg analyses")


# ── Helper: extract series ─────────────────────────────────────────
def series(logs, key):
    epochs = [d["epoch"] for d in logs]
    vals = [d.get(key, 0) for d in logs]
    return np.array(epochs), np.array(vals)


# ====================================================================
# FIGURE 1: Survival Rate Comparison (all 3 runs)
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for run_name in RUN_FILES:
    evals = all_data[run_name]["evals"]
    epochs = [b["epoch"] for b in evals]
    survival = [b["survival"] for b in evals]
    ax.plot(epochs, survival, marker=RUN_MARKERS[run_name], linestyle="-",
            color=RUN_COLORS[run_name], linewidth=2, markersize=7,
            label=RUN_LABELS[run_name], zorder=5)

ax.axhline(y=BASELINE_RANDOM, color="#e74c3c", linestyle="--", linewidth=1.5,
           alpha=0.6, label=f"Random baseline ({BASELINE_RANDOM}%)")
ax.axhline(y=BASELINE_GREEDY, color="#3498db", linestyle=":", linewidth=1.5,
           alpha=0.6, label=f"Greedy baseline ({BASELINE_GREEDY}%)")
ax.axhline(y=BASELINE_OPTIMAL, color="#27ae60", linestyle="--", linewidth=1.5,
           alpha=0.4, label=f"Optimal baseline ({BASELINE_OPTIMAL}%)")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Survival Rate (%)", fontsize=13)
ax.set_title("Survival Rate Comparison — Runs 5, 6, 7", fontsize=14, pad=15)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_ylim(15, 65)
ax.set_xlim(0, 75)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "01_survival_rate_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 01_survival_rate_comparison.png")


# ====================================================================
# FIGURE 2: Mean Reward from Evaluation Episodes
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for run_name in RUN_FILES:
    evals = all_data[run_name]["evals"]
    epochs = [b["epoch"] for b in evals]
    rewards = [b["mean_reward"] for b in evals]
    ax.plot(epochs, rewards, marker=RUN_MARKERS[run_name], linestyle="-",
            color=RUN_COLORS[run_name], linewidth=2, markersize=7,
            label=RUN_LABELS[run_name])

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Mean Reward (100 episodes)", fontsize=13)
ax.set_title("Evaluation Mean Reward — Runs 5, 6, 7", fontsize=14, pad=15)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
ax.set_xlim(0, 75)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "02_mean_reward_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 02_mean_reward_comparison.png")


# ====================================================================
# FIGURE 3: Reconstruction Accuracy Over Training (test set)
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 5))

for run_name in RUN_FILES:
    test_data = all_data[run_name]["test"]
    ep, acc = series(test_data, "recon_acc")
    ax.plot(ep, acc * 100, color=RUN_COLORS[run_name], linewidth=1.5,
            alpha=0.8, label=RUN_LABELS[run_name])

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Reconstruction Accuracy (%)", fontsize=13)
ax.set_title("Entity Reconstruction Accuracy (Test Set) — Runs 5, 6, 7", fontsize=14, pad=15)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_ylim(0, 105)
ax.set_xlim(0, 75)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "03_recon_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 03_recon_accuracy.png")


# ====================================================================
# FIGURE 4: Action Distribution at Final Epoch (grouped bar chart)
# ====================================================================
actions = ["hunt", "gather", "flee", "rest", "eat",
           "craft_spear", "craft_fire", "craft_shelter", "craft_rod",
           "mitigate", "endure"]
action_labels = ["Hunt", "Gather", "Flee", "Rest", "Eat",
                 "Spear", "Fire", "Shelter", "Rod",
                 "Mitigate", "Endure"]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(actions))
width = 0.25

for idx, run_name in enumerate(RUN_FILES):
    evals = all_data[run_name]["evals"]
    final = evals[-1]  # last evaluation
    vals = [final.get(f"eval_{a}", 0) for a in actions]
    bars = ax.bar(x + (idx - 1) * width, vals, width,
                  color=RUN_COLORS[run_name], alpha=0.85,
                  label=f"{run_name} (ep {final['epoch']})")

ax.set_xlabel("Action", fontsize=13)
ax.set_ylabel("Rate (%)", fontsize=13)
ax.set_title("Action Distribution at Final Evaluation — Runs 5, 6, 7", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(action_labels, fontsize=10, rotation=30, ha="right")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "04_action_distribution_final.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 04_action_distribution_final.png")


# ====================================================================
# FIGURE 5: Action Distribution Evolution (per run, stacked area)
# ====================================================================
for run_name in RUN_FILES:
    fig, ax = plt.subplots(figsize=(12, 6))
    evals = all_data[run_name]["evals"]
    epochs = [b["epoch"] for b in evals]

    action_keys = ["hunt", "gather", "flee", "rest", "eat",
                   "craft_spear", "craft_fire", "craft_shelter", "craft_rod",
                   "mitigate"]
    action_display = ["Hunt", "Gather", "Flee", "Rest", "Eat",
                      "Craft Spear", "Craft Fire", "Craft Shelter", "Craft Rod",
                      "Mitigate"]
    bar_colors = ["#e41a1c", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00",
                  "#a65628", "#c4803c", "#d4a05c", "#e4c08c", "#999999"]

    bottom = np.zeros(len(epochs))
    for action, label, color in zip(action_keys, action_display, bar_colors):
        vals = np.array([b.get(f"eval_{action}", 0) for b in evals])
        ax.bar(epochs, vals, bottom=bottom, width=3.5, color=color, alpha=0.85, label=label)
        bottom += vals

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Action Rate (%)", fontsize=13)
    ax.set_title(f"Action Distribution Over Training — {run_name}", fontsize=14, pad=15)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9)
    ax.set_xlim(0, max(epochs) + 5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    safe_name = run_name.lower().replace(" ", "")
    fig.savefig(FIG_DIR / f"05_action_evolution_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_action_evolution_{safe_name}.png")


# ====================================================================
# FIGURE 6: Message Diversity Over Training
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

entity_types = ["Animal", "Resource", "Danger", "CraftOpp", "Event"]
entity_colors = ["#e41a1c", "#4daf4a", "#377eb8", "#ff7f00", "#984ea3"]

# Left: Total unique messages per run
ax = axes[0]
for run_name in RUN_FILES:
    msgs = all_data[run_name]["msgs"]
    epochs = [m["epoch"] for m in msgs]
    totals = [sum(m["entities"].get(et, {}).get("unique", 0) for et in entity_types)
              for m in msgs]
    ax.plot(epochs, totals, marker=RUN_MARKERS[run_name], linestyle="-",
            color=RUN_COLORS[run_name], linewidth=2, markersize=7,
            label=RUN_LABELS[run_name])

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Total Unique Messages", fontsize=12)
ax.set_title("Total Unique Messages Over Training", fontsize=13, pad=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.set_ylim(0, 45)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

# Right: Per-entity-type unique messages at final analysis
ax = axes[1]
x = np.arange(len(entity_types))
width = 0.25

for idx, run_name in enumerate(RUN_FILES):
    msgs = all_data[run_name]["msgs"]
    final_msg = msgs[-1]
    vals = [final_msg["entities"].get(et, {}).get("unique", 0) for et in entity_types]
    ax.bar(x + (idx - 1) * width, vals, width,
           color=RUN_COLORS[run_name], alpha=0.85,
           label=f"{run_name} (ep {final_msg['epoch']})")

ax.set_xlabel("Entity Type", fontsize=12)
ax.set_ylabel("Unique Messages", fontsize=12)
ax.set_title("Unique Messages per Entity Type — Final Epoch", fontsize=13, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(entity_types, fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "06_message_diversity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 06_message_diversity.png")


# ====================================================================
# FIGURE 7: Training Loss and Expected Reward (train set)
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Expected reward (train)
ax = axes[0]
for run_name in RUN_FILES:
    train_data = all_data[run_name]["train"]
    ep, exp_rew = series(train_data, "expected_reward")
    ax.plot(ep, exp_rew, color=RUN_COLORS[run_name], linewidth=1.5,
            alpha=0.8, label=RUN_LABELS[run_name])

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Expected Reward (train)", fontsize=12)
ax.set_title("Expected Reward During Training", fontsize=13, pad=10)
ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

# Right: Recon loss (train)
ax = axes[1]
for run_name in RUN_FILES:
    train_data = all_data[run_name]["train"]
    ep, rloss = series(train_data, "recon_loss")
    ax.plot(ep, rloss, color=RUN_COLORS[run_name], linewidth=1.5,
            alpha=0.8, label=RUN_LABELS[run_name])

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Reconstruction Loss (train)", fontsize=12)
ax.set_title("Reconstruction Loss During Training", fontsize=13, pad=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "07_training_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 07_training_metrics.png")


# ====================================================================
# FIGURE 8: Action Rate Trends (train set, key actions over epochs)
# ====================================================================
key_actions = ["hunt_rate", "gather_rate", "flee_rate", "eat_rate", "rest_rate"]
key_labels = ["Hunt", "Gather", "Flee", "Eat", "Rest"]
key_colors = ["#e41a1c", "#4daf4a", "#377eb8", "#ff7f00", "#984ea3"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax_idx, run_name in enumerate(RUN_FILES):
    ax = axes[ax_idx]
    train_data = all_data[run_name]["train"]

    for action, label, color in zip(key_actions, key_labels, key_colors):
        ep, vals = series(train_data, action)
        ax.plot(ep, vals * 100, color=color, linewidth=1.5, alpha=0.8, label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    if ax_idx == 0:
        ax.set_ylabel("Action Rate (%)", fontsize=12)
    ax.set_title(f"{run_name}", fontsize=13, pad=10)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

fig.suptitle("Key Action Rates Over Training (Train Set)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "08_action_rates_train.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 08_action_rates_train.png")


# ====================================================================
# FIGURE 9: Summary Statistics Table (as figure)
# ====================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

# Compute summary stats
rows = []
for run_name in RUN_FILES:
    evals = all_data[run_name]["evals"]
    msgs = all_data[run_name]["msgs"]
    test_data = all_data[run_name]["test"]

    survivals = [b["survival"] for b in evals]
    rewards = [b["mean_reward"] for b in evals]
    final_msg = msgs[-1] if msgs else {}
    total_unique = sum(final_msg.get("entities", {}).get(et, {}).get("unique", 0)
                       for et in entity_types)
    final_recon = test_data[-1].get("recon_acc", 0) * 100 if test_data else 0

    rows.append([
        run_name,
        f"{np.mean(survivals):.1f}%",
        f"{np.max(survivals):.0f}%",
        f"{np.min(survivals):.0f}%",
        f"{np.std(survivals):.1f}",
        f"{np.mean(rewards):.1f}",
        f"{total_unique}",
        f"{final_recon:.1f}%",
    ])

col_labels = ["Run", "Mean Surv.", "Peak", "Min", "Std", "Mean Reward", "Unique Msgs", "Final Recon"]
table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                 cellLoc="center", colColours=["#d5e8d4"] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.auto_set_column_width(col=list(range(len(col_labels))))
table.scale(1, 1.8)

# Color run name cells
for i, run_name in enumerate(RUN_FILES):
    table[i + 1, 0].set_facecolor(RUN_COLORS[run_name])
    table[i + 1, 0].set_text_props(color="white", fontweight="bold")

ax.set_title("Summary Statistics — Runs 5, 6, 7", fontsize=14, pad=20)

fig.tight_layout()
fig.savefig(FIG_DIR / "09_summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 09_summary_table.png")


# ====================================================================
# FIGURE 10: Survival Bar Chart with Error Context
# ====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

run_names = list(RUN_FILES.keys())
means = []
stds = []
peaks = []

for run_name in run_names:
    survivals = [b["survival"] for b in all_data[run_name]["evals"]]
    means.append(np.mean(survivals))
    stds.append(np.std(survivals))
    peaks.append(np.max(survivals))

x = np.arange(len(run_names))
bars = ax.bar(x, means, color=[RUN_COLORS[r] for r in run_names],
              alpha=0.85, edgecolor="white", linewidth=1.5)
ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black",
            capsize=8, capthick=2, linewidth=2)

# Mark peaks
for i, (peak, mean) in enumerate(zip(peaks, means)):
    ax.plot(i, peak, marker="*", color="gold", markersize=16, zorder=10,
            markeredgecolor="black", markeredgewidth=0.5)
    ax.annotate(f"{peak:.0f}%", xy=(i, peak), xytext=(i + 0.15, peak + 1),
                fontsize=10, fontweight="bold", color="#2c3e50")

ax.axhline(y=BASELINE_RANDOM, color="#e74c3c", linestyle="--", linewidth=1.5,
           alpha=0.6, label=f"Random ({BASELINE_RANDOM}%)")
ax.axhline(y=BASELINE_GREEDY, color="#3498db", linestyle=":", linewidth=1.5,
           alpha=0.5, label=f"Greedy ({BASELINE_GREEDY}%)")

ax.set_xticks(x)
ax.set_xticklabels(run_names, fontsize=12, fontweight="bold")
ax.set_ylabel("Survival Rate (%)", fontsize=13)
ax.set_title("Mean Survival Rate ± Std Dev (★ = peak)", fontsize=14, pad=15)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.2, axis="y")
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "10_survival_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 10_survival_summary_bar.png")


print(f"\nAll figures saved to: {FIG_DIR}")
