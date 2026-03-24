#!/usr/bin/env python3
"""Compact comparison for run11 (decay) vs run12 (no-decay).

Outputs:
- run11_vs_run12_summary.csv
- run11_vs_run12_total_unique.png
- run11_vs_run12_unique_by_type.png
- run11_vs_run12_final_by_type.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


TYPE_NAME = {
    "0": "Animals",
    "1": "Resources",
    "2": "Threats",
    "3": "Tools",
    "4": "Terrain",
}


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda x: int(x["epoch"]))
    return records


def load_final_snapshot(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_progression(records: List[dict]) -> Dict[str, float]:
    totals = [int(r["total_unique_messages"]) for r in records]
    epochs = [int(r["epoch"]) for r in records]
    epochs_at_40 = [e for e, t in zip(epochs, totals) if t == 40]

    return {
        "checkpoints": float(len(records)),
        "first_epoch_at_40": float(epochs_at_40[0]) if epochs_at_40 else -1.0,
        "num_checkpoints_at_40": float(len(epochs_at_40)),
        "fraction_checkpoints_at_40": float(len(epochs_at_40)) / len(records) if records else 0.0,
        "final_total_unique": float(totals[-1]) if totals else 0.0,
        "mean_total_unique": mean(totals) if totals else 0.0,
    }


def parse_last_test_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    keys = ["recon_acc", "mean_reward", "hunt_rate", "gather_rate", "craft_rate"]
    wanted = {k: None for k in keys}

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("mode") != "test":
            continue
        # Keep only epoch-stat dictionaries; skip entropy/posdis entries.
        if "recon_acc" not in obj:
            continue
        for key in keys:
            if key in obj:
                wanted[key] = float(obj[key])
        break

    return wanted


def plot_total_unique(run11: List[dict], run12: List[dict], out_path: Path) -> None:
    e11 = [int(r["epoch"]) for r in run11]
    t11 = [int(r["total_unique_messages"]) for r in run11]
    e12 = [int(r["epoch"]) for r in run12]
    t12 = [int(r["total_unique_messages"]) for r in run12]

    plt.figure(figsize=(8, 4.5))
    plt.plot(e11, t11, marker="o", linewidth=2, label="Run11 decay=0.99")
    plt.plot(e12, t12, marker="o", linewidth=2, label="Run12 decay=1.0")
    plt.axhline(40, linestyle="--", linewidth=1, color="gray", label="Target = 40")
    plt.title("Total Unique Messages Over Training")
    plt.xlabel("Epoch")
    plt.ylabel("Unique messages")
    plt.ylim(0, 42)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_unique_by_type(run11: List[dict], run12: List[dict], out_path: Path) -> None:
    epochs11 = [int(r["epoch"]) for r in run11]
    epochs12 = [int(r["epoch"]) for r in run12]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)

    for idx, type_id in enumerate(["0", "1", "2", "3", "4"]):
        vals11 = [int(r["by_type"][type_id]["unique"]) for r in run11]
        vals12 = [int(r["by_type"][type_id]["unique"]) for r in run12]

        ax = axes[idx]
        ax.plot(epochs11, vals11, marker="o", linewidth=1.8, label="Run11")
        ax.plot(epochs12, vals12, marker="o", linewidth=1.8, label="Run12")
        ax.set_title(TYPE_NAME[type_id])
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Unique messages")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Unique Messages by Type")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_by_type(final11: dict, final12: dict, out_path: Path) -> None:
    type_ids = ["0", "1", "2", "3", "4"]
    labels = [TYPE_NAME[t] for t in type_ids]
    vals11 = [int(final11["by_type"][t]["unique"]) for t in type_ids]
    vals12 = [int(final12["by_type"][t]["unique"]) for t in type_ids]

    x = list(range(len(type_ids)))
    width = 0.38

    plt.figure(figsize=(8, 4.5))
    plt.bar([p - width / 2 for p in x], vals11, width=width, label="Run11 decay=0.99")
    plt.bar([p + width / 2 for p in x], vals12, width=width, label="Run12 decay=1.0")
    plt.xticks(x, labels)
    plt.ylabel("Final unique messages")
    plt.title("Final Type-Level Lexicon Size")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_summary_csv(
    out_path: Path,
    summary11: Dict[str, float],
    summary12: Dict[str, float],
    metrics11: Dict[str, Optional[float]],
    metrics12: Dict[str, Optional[float]],
) -> None:
    rows = [
        ["metric", "run11_decay_0.99", "run12_decay_1.0"],
        ["checkpoints", summary11["checkpoints"], summary12["checkpoints"]],
        ["first_epoch_at_40", summary11["first_epoch_at_40"], summary12["first_epoch_at_40"]],
        ["num_checkpoints_at_40", summary11["num_checkpoints_at_40"], summary12["num_checkpoints_at_40"]],
        ["fraction_checkpoints_at_40", summary11["fraction_checkpoints_at_40"], summary12["fraction_checkpoints_at_40"]],
        ["mean_total_unique", summary11["mean_total_unique"], summary12["mean_total_unique"]],
        ["final_total_unique", summary11["final_total_unique"], summary12["final_total_unique"]],
        ["final_test_recon_acc", metrics11["recon_acc"], metrics12["recon_acc"]],
        ["final_test_mean_reward", metrics11["mean_reward"], metrics12["mean_reward"]],
        ["final_test_hunt_rate", metrics11["hunt_rate"], metrics12["hunt_rate"]],
        ["final_test_gather_rate", metrics11["gather_rate"], metrics12["gather_rate"]],
        ["final_test_craft_rate", metrics11["craft_rate"], metrics12["craft_rate"]],
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def build_default_paths() -> Dict[str, Path]:
    dissertation_root = Path(__file__).resolve().parents[2]
    outputs = dissertation_root / "outputs"
    return {
        "run11_progression": outputs / "message_progression_run11.jsonl",
        "run12_progression": outputs / "message_progression_run12.jsonl",
        "run11_snapshot": outputs / "message_snapshot_final_run11.json",
        "run12_snapshot": outputs / "message_snapshot_final_run12.json",
        "run11_log": outputs / "train_run11.log",
        "run12_log": outputs / "train_run12.log",
        "out_dir": outputs,
    }


def parse_args() -> argparse.Namespace:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Compare run11 vs run12 language progression")
    parser.add_argument("--run11_progression", type=Path, default=defaults["run11_progression"])
    parser.add_argument("--run12_progression", type=Path, default=defaults["run12_progression"])
    parser.add_argument("--run11_snapshot", type=Path, default=defaults["run11_snapshot"])
    parser.add_argument("--run12_snapshot", type=Path, default=defaults["run12_snapshot"])
    parser.add_argument("--run11_log", type=Path, default=defaults["run11_log"])
    parser.add_argument("--run12_log", type=Path, default=defaults["run12_log"])
    parser.add_argument("--out_dir", type=Path, default=defaults["out_dir"])
    return parser.parse_args()


def validate_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def main() -> None:
    args = parse_args()

    for p in [
        args.run11_progression,
        args.run12_progression,
        args.run11_snapshot,
        args.run12_snapshot,
        args.run11_log,
        args.run12_log,
    ]:
        validate_exists(p)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run11 = load_jsonl(args.run11_progression)
    run12 = load_jsonl(args.run12_progression)
    final11 = load_final_snapshot(args.run11_snapshot)
    final12 = load_final_snapshot(args.run12_snapshot)

    summary11 = summarize_progression(run11)
    summary12 = summarize_progression(run12)
    metrics11 = parse_last_test_metrics(args.run11_log)
    metrics12 = parse_last_test_metrics(args.run12_log)

    summary_csv = args.out_dir / "run11_vs_run12_summary.csv"
    fig_total = args.out_dir / "run11_vs_run12_total_unique.png"
    fig_by_type = args.out_dir / "run11_vs_run12_unique_by_type.png"
    fig_final_type = args.out_dir / "run11_vs_run12_final_by_type.png"

    write_summary_csv(summary_csv, summary11, summary12, metrics11, metrics12)
    plot_total_unique(run11, run12, fig_total)
    plot_unique_by_type(run11, run12, fig_by_type)
    plot_final_by_type(final11, final12, fig_final_type)

    print("Wrote:")
    print(f"  - {summary_csv}")
    print(f"  - {fig_total}")
    print(f"  - {fig_by_type}")
    print(f"  - {fig_final_type}")


if __name__ == "__main__":
    main()
