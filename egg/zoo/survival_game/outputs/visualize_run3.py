#!/usr/bin/env python3
"""
Visualize training results from run3.txt (Death Proximity Penalty experiment).
Generates publication-quality figures for supervisor presentation.
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
OUTPUT_FILE = Path(__file__).parent / "outputs" / "run3.txt"
FIG_DIR = Path(__file__).parent / "outputs" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colour palette (colour-blind friendly)
COLORS = {
    "hunt": "#e41a1c",
    "gather": "#4daf4a",
    "flee": "#377eb8",
    "rest": "#984ea3",
    "eat": "#ff7f00",
    "craft": "#a65628",
    "mitigate": "#999999",
    "endure": "#f781bf",
}

# ── Parse the log ─────────────────────────────────────────────────
train_logs = []
test_logs = []
eval_blocks = []

with open(OUTPUT_FILE) as f:
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
    i += 1

# ── Helper: extract series ─────────────────────────────────────────
def series(logs, key):
    epochs = [d["epoch"] for d in logs]
    vals = [d.get(key, 0) for d in logs]
    return np.array(epochs), np.array(vals)

# Baselines from prototype.py
BASELINE_RANDOM = 21.1
BASELINE_GREEDY = 55.5
BASELINE_OPTIMAL = 63.1

# ── Figure 1: Survival Rate Over Training ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
eval_epochs = [b["epoch"] for b in eval_blocks]
eval_survival = [b["survival"] for b in eval_blocks]

ax.plot(eval_epochs, eval_survival, "o-", color="#2c3e50", linewidth=2.5,
        markersize=8, label="Agent (Death Penalty)", zorder=5)
ax.axhline(y=BASELINE_RANDOM, color="#e74c3c", linestyle="--", linewidth=1.5,
           alpha=0.7, label=f"Random baseline ({BASELINE_RANDOM}%)")
ax.axhline(y=BASELINE_OPTIMAL, color="#27ae60", linestyle="--", linewidth=1.5,
           alpha=0.7, label=f"Optimal baseline ({BASELINE_OPTIMAL}%)")

ax.fill_between(eval_epochs, BASELINE_RANDOM, eval_survival,
                alpha=0.15, color="#3498db", label="Gain over random")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Survival Rate (%)", fontsize=13)
ax.set_title("Survival Rate During Training — Run 3 (Death Proximity Penalty)", fontsize=14, pad=15)
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, 70)
ax.set_xlim(0, 105)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# Annotate peak
peak_idx = np.argmax(eval_survival)
ax.annotate(f"Peak: {eval_survival[peak_idx]}%",
            xy=(eval_epochs[peak_idx], eval_survival[peak_idx]),
            xytext=(eval_epochs[peak_idx] + 8, eval_survival[peak_idx] + 5),
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2c3e50"),
            color="#2c3e50")

fig.tight_layout()
fig.savefig(FIG_DIR / "survival_rate.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'survival_rate.png'}")


# ── Figure 2: Action Distribution Over Training (Train mode) ──────
fig, ax = plt.subplots(figsize=(12, 6))

actions = ["hunt", "gather", "flee", "rest", "eat", "craft", "mitigate"]
action_labels = ["Hunt", "Gather", "Flee", "Rest", "Eat", "Craft", "Mitigate"]

for action, label in zip(actions, action_labels):
    key = f"{action}_rate"
    epochs, vals = series(train_logs, key)
    ax.plot(epochs, vals * 100, label=label, color=COLORS[action],
            linewidth=2 if action in ("eat", "hunt", "gather") else 1.2,
            alpha=1.0 if action in ("eat", "hunt", "gather") else 0.7)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Action Rate (%)", fontsize=13)
ax.set_title("Action Distribution During Training (Train Set)", fontsize=14, pad=15)
ax.legend(loc="right", fontsize=10, framealpha=0.9)
ax.set_xlim(0, 105)
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "action_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'action_distribution.png'}")


# ── Figure 3: Evaluation Action Distribution (Stacked Bar) ────────
fig, ax = plt.subplots(figsize=(12, 6))

eval_actions = ["hunt", "gather", "flee", "rest", "eat",
                "craft_spear", "craft_fire", "craft_shelter", "craft_rod",
                "mitigate", "endure"]
eval_action_labels = ["Hunt", "Gather", "Flee", "Rest", "Eat",
                       "Craft Spear", "Craft Fire", "Craft Shelter", "Craft Rod",
                       "Mitigate", "Endure"]
bar_colors = ["#e41a1c", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00",
              "#a65628", "#c4803c", "#d4a05c", "#e4c08c",
              "#999999", "#f781bf"]

bottom = np.zeros(len(eval_blocks))
x = np.arange(len(eval_blocks))
bar_width = 0.7

for action, label, color in zip(eval_actions, eval_action_labels, bar_colors):
    vals = [b.get(f"eval_{action}", 0) for b in eval_blocks]
    ax.bar(x, vals, bar_width, bottom=bottom, label=label, color=color,
           edgecolor="white", linewidth=0.5)
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([f"Ep {b['epoch']}" for b in eval_blocks], fontsize=10)
ax.set_ylabel("Action Distribution (%)", fontsize=13)
ax.set_title("Evaluation Action Distribution per Epoch", fontsize=14, pad=15)
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=9)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.2, axis="y")

fig.tight_layout()
fig.savefig(FIG_DIR / "eval_action_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'eval_action_stacked.png'}")


# ── Figure 4: Loss and Reward Curves ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Loss
for mode, logs, color, ls in [("Train", train_logs, "#2c3e50", "-"),
                                ("Test", test_logs, "#e74c3c", "--")]:
    ep, vals = series(logs, "loss")
    axes[0].plot(ep, vals, ls, color=color, label=mode, linewidth=1.5)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Total Loss", fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Mean Reward
for mode, logs, color, ls in [("Train", train_logs, "#2c3e50", "-"),
                                ("Test", test_logs, "#e74c3c", "--")]:
    ep, vals = series(logs, "mean_reward")
    axes[1].plot(ep, vals, ls, color=color, label=mode, linewidth=1.5)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Mean Reward", fontsize=12)
axes[1].set_title("Mean Expected Reward per Step", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Recon Loss (log scale)
for mode, logs, color, ls in [("Train", train_logs, "#2c3e50", "-"),
                                ("Test", test_logs, "#e74c3c", "--")]:
    ep, vals = series(logs, "recon_loss")
    axes[2].semilogy(ep, vals + 1e-10, ls, color=color, label=mode, linewidth=1.5)
axes[2].set_xlabel("Epoch", fontsize=12)
axes[2].set_ylabel("Recon Loss (log scale)", fontsize=12)
axes[2].set_title("Reconstruction Loss", fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

fig.suptitle("Training Curves — Run 3", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'training_curves.png'}")


# ── Figure 5: Reconstruction Accuracy ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for mode, logs, color, ls in [("Train", train_logs, "#2c3e50", "-"),
                                ("Test", test_logs, "#e74c3c", "--")]:
    ep, vals = series(logs, "recon_acc")
    ax.plot(ep, vals * 100, ls, color=color, label=mode, linewidth=2)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Reconstruction Accuracy (%)", fontsize=13)
ax.set_title("Receiver Reconstruction Accuracy (Entity Type Recovery)", fontsize=14, pad=15)
ax.legend(fontsize=11)
ax.set_ylim(60, 101)
ax.set_xlim(0, 105)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "recon_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'recon_accuracy.png'}")


# ── Figure 6: Eat Rate — Before vs After Death Penalty ────────────
fig, ax = plt.subplots(figsize=(10, 5))

ep_train, eat_train = series(train_logs, "eat_rate")
ep_test, eat_test = series(test_logs, "eat_rate")

ax.plot(ep_train, eat_train * 100, "-", color="#ff7f00", linewidth=2.5,
        label="Eat Rate (train)", zorder=5)
ax.plot(ep_test, eat_test * 100, "--", color="#ff7f00", linewidth=1.5,
        alpha=0.7, label="Eat Rate (test)")

# Annotate the critical improvement
ax.annotate("0% in ALL previous runs\n(before death penalty)",
            xy=(1, 31.4), xytext=(15, 37),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=2),
            color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.8))

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Eat Action Rate (%)", fontsize=13)
ax.set_title("Eat Rate — Death Proximity Penalty Breakthrough", fontsize=14, pad=15)
ax.legend(fontsize=11)
ax.set_xlim(0, 105)
ax.set_ylim(0, 40)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "eat_rate_breakthrough.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'eat_rate_breakthrough.png'}")


# ── Figure 7: Message Evolution Summary ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

msg_data = [
    ["Epoch", "Animal", "Resource", "Danger", "CraftOpp", "Event"],
    ["10", "[25 25 25]×580\n[4 25 25]×234", "[4 4 4]×1105", "[4 7 7]×695", "[7 7 7]×351", "[3 3 3]×323"],
    ["30", "[25 25 25]×580\n[4 25 25]×234", "[4 4 4]×1105", "[37 7 7]×695", "[7 7 7]×351", "[3 3 3]×323"],
    ["60", "[25 25 25]×814", "[4 4 4]×1105", "[37 7 7]×500\n[4 7 7]×195", "[7 7 7]×351", "[3 3 3]×323"],
    ["90", "[25 25 25]×580\n[4 25 25]×234", "[4 4 4]×1105", "[37 20 7]×695", "[7 7 7]×351", "[3 3 3]×257\n[37 7 7]×66"],
    ["100", "[25 25 25]×580\n[4 25 25]×234", "[4 4 4]×1105", "[37 37 10]×500\n[37 4 7]×195", "[7 7 7]×351", "[3 3 3]×257\n[37 7 7]×66"],
]

table = ax.table(cellText=msg_data[1:], colLabels=msg_data[0],
                 cellLoc="center", loc="center",
                 colWidths=[0.07, 0.22, 0.15, 0.22, 0.14, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Colour header
for j in range(6):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight evolving cells (Danger column = col 3)
for row in range(1, 6):
    table[row, 3].set_facecolor("#ffeaa7")  # Danger evolves
    if row >= 4:
        table[row, 5].set_facecolor("#dfe6e9")  # Event splits

ax.set_title("Message Protocol Evolution\n(Danger messages evolve throughout training — highlighted in yellow)",
             fontsize=13, pad=20)

fig.tight_layout()
fig.savefig(FIG_DIR / "message_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'message_evolution.png'}")


# ── Figure 8: Summary Dashboard ───────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Panel A: Survival rate
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(eval_epochs, eval_survival, "o-", color="#2c3e50", linewidth=2, markersize=6)
ax1.axhline(BASELINE_RANDOM, color="#e74c3c", ls="--", lw=1, alpha=0.6)
ax1.axhline(BASELINE_OPTIMAL, color="#27ae60", ls="--", lw=1, alpha=0.6)
ax1.set_title("(A) Survival Rate", fontsize=12, fontweight="bold")
ax1.set_ylabel("%")
ax1.set_ylim(0, 70)
ax1.grid(True, alpha=0.3)
ax1.text(80, BASELINE_RANDOM + 2, "Random", fontsize=8, color="#e74c3c")
ax1.text(80, BASELINE_OPTIMAL + 2, "Optimal", fontsize=8, color="#27ae60")

# Panel B: Key actions evolution
ax2 = fig.add_subplot(gs[0, 1])
for action, label, color in [("eat", "Eat", "#ff7f00"), ("hunt", "Hunt", "#e41a1c"),
                               ("gather", "Gather", "#4daf4a"), ("flee", "Flee", "#377eb8")]:
    ep, vals = series(train_logs, f"{action}_rate")
    ax2.plot(ep, vals * 100, color=color, label=label, linewidth=1.5)
ax2.set_title("(B) Key Action Rates (Train)", fontsize=12, fontweight="bold")
ax2.set_ylabel("%")
ax2.legend(fontsize=8, loc="right")
ax2.grid(True, alpha=0.3)

# Panel C: Recon accuracy
ax3 = fig.add_subplot(gs[0, 2])
ep, vals = series(train_logs, "recon_acc")
ax3.plot(ep, vals * 100, color="#8e44ad", linewidth=2)
ax3.set_title("(C) Recon Accuracy", fontsize=12, fontweight="bold")
ax3.set_ylabel("%")
ax3.set_ylim(60, 101)
ax3.grid(True, alpha=0.3)

# Panel D: Loss curve
ax4 = fig.add_subplot(gs[1, 0])
ep, vals = series(train_logs, "loss")
ax4.plot(ep, vals, color="#2c3e50", linewidth=1.5)
ax4.set_title("(D) Training Loss", fontsize=12, fontweight="bold")
ax4.set_ylabel("Loss")
ax4.set_xlabel("Epoch")
ax4.grid(True, alpha=0.3)

# Panel E: Mean reward
ax5 = fig.add_subplot(gs[1, 1])
for mode, logs, color, ls in [("Train", train_logs, "#2c3e50", "-"),
                                ("Test", test_logs, "#e74c3c", "--")]:
    ep, vals = series(logs, "mean_reward")
    ax5.plot(ep, vals, ls, color=color, label=mode, linewidth=1.5)
ax5.set_title("(E) Mean Reward per Step", fontsize=12, fontweight="bold")
ax5.set_xlabel("Epoch")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Panel F: Summary statistics text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

summary_text = (
    "Run 3 Summary Statistics\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Peak Survival:    {max(eval_survival):.0f}%  (Ep {eval_epochs[np.argmax(eval_survival)]})\n"
    f"Final Survival:   {eval_survival[-1]:.0f}%  (Ep 100)\n"
    f"Mean Survival:    {np.mean(eval_survival):.1f}%\n"
    f"Random Baseline:  {BASELINE_RANDOM}%\n"
    f"Optimal Baseline: {BASELINE_OPTIMAL}%\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Final Eat Rate:   {train_logs[-1]['eat_rate']*100:.1f}%\n"
    f"Final Hunt Rate:  {train_logs[-1]['hunt_rate']*100:.1f}%\n"
    f"Final Gather:     {train_logs[-1]['gather_rate']*100:.1f}%\n"
    f"Recon Accuracy:   {train_logs[-1]['recon_acc']*100:.1f}%\n"
    f"Distinct Msgs:    5 entity-specific\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Key Achievement:\n"
    "  Eat rate: 0% → 18.6%\n"
    "  (death penalty enabled\n"
    "   survival-critical eating)"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
ax6.set_title("(F) Key Statistics", fontsize=12, fontweight="bold")

fig.suptitle("Survival Game — Run 3 Training Dashboard\n"
             "(Death Proximity Penalty + recon_weight=2.0, reward_scale=0.2)",
             fontsize=15, fontweight="bold", y=1.02)
fig.savefig(FIG_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'dashboard.png'}")


# ── Print summary to terminal ──────────────────────────────────────
print("\n" + "=" * 60)
print("  Run 3 Analysis Summary")
print("=" * 60)
print(f"  Epochs:              100")
print(f"  Peak Survival:       {max(eval_survival):.0f}% (Epoch {eval_epochs[np.argmax(eval_survival)]})")
print(f"  Mean Survival:       {np.mean(eval_survival):.1f}%")
print(f"  Final Survival:      {eval_survival[-1]:.0f}%")
print(f"  Random Baseline:     {BASELINE_RANDOM}%")
print(f"  Optimal Baseline:    {BASELINE_OPTIMAL}%")
print(f"  Gain over Random:    +{np.mean(eval_survival) - BASELINE_RANDOM:.1f}pp average")
print(f"  Communication Gap:   {BASELINE_OPTIMAL - np.mean(eval_survival):.1f}pp to optimal")
print(f"  Final Train Loss:    {train_logs[-1]['loss']:.3f}")
print(f"  Final Recon Acc:     {train_logs[-1]['recon_acc']*100:.1f}%")
print(f"  Final Recon Loss:    {train_logs[-1]['recon_loss']:.2e}")
print(f"  Distinct Messages:   5 entity-specific patterns")
print(f"  Msg Evolution:       Danger messages evolved 4 times")
print(f"")
print(f"  Key Breakthrough:")
print(f"    Eat rate went from 0% (all prior runs) → ~19% (run3)")
print(f"    Death proximity penalty successfully enabled eating")
print(f"    behaviour, solving the starvation death spiral.")
print("=" * 60)
print(f"\n  All figures saved to: {FIG_DIR}/")
