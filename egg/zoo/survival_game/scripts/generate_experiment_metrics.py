#!/usr/bin/env python3
"""
Generate experiment-level metrics for survival_game outputs.

This script is intentionally scoped for dissertation analysis where runs are grouped
by experiment folders. It computes averages across runs within each experiment,
exports message reuse tables, and generates cross-experiment TopSim comparisons.

Design constraints:
- No Hungarian analysis
- No message/entity heatmaps
- No mixed 15-run tuning aggregate; each tuning sub-folder is handled separately
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _extract_run_id(name: str) -> Optional[int]:
    match = re.search(r"run(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def _stable_run_label(file_name: str) -> str:
    stem = Path(file_name).stem
    run_id = _extract_run_id(stem)
    if run_id is not None:
        return f"run{run_id}"
    return stem


def _sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=np.float64), ddof=1))


def _discover_logs(run_dir: Path, pattern: str) -> List[Path]:
    candidates = sorted(run_dir.glob(pattern))
    files = [p for p in candidates if p.is_file() and p.suffix in {".txt", ".log"}]
    if files:
        return files

    if pattern.endswith(".txt") or pattern.endswith(".log"):
        return []

    # Stem fallback for convenience.
    txt = sorted(run_dir.glob(f"{pattern}.txt"))
    log = sorted(run_dir.glob(f"{pattern}.log"))
    merged = {p.name: p for p in txt + log}
    return [merged[k] for k in sorted(merged)]


def _parse_log_file(log_file: Path) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], List[int]]:
    train_metrics: Dict[int, Dict[str, float]] = {}
    test_metrics: Dict[int, Dict[str, float]] = {}
    epochs = set()

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = payload.get("mode")
            epoch = payload.get("epoch")
            if mode not in {"train", "test"} or epoch is None:
                continue

            epoch = int(epoch)
            target = train_metrics if mode == "train" else test_metrics
            target.setdefault(epoch, {})
            for key, value in payload.items():
                if key in {"mode", "epoch"}:
                    continue
                try:
                    target[epoch][key] = float(value)
                except (TypeError, ValueError):
                    continue
            epochs.add(epoch)

    return {"train": train_metrics, "test": test_metrics}, sorted(epochs)


def _load_snapshot_for_log(run_dir: Path, log_file: Path) -> Optional[Dict]:
    run_id = _extract_run_id(log_file.stem)
    candidates: List[Path] = []

    if run_id is not None:
        candidates.append(run_dir / f"message_snapshot_final_run{run_id}.json")

    candidates.extend(sorted(run_dir.glob(f"message_snapshot_final*{log_file.stem}*.json")))
    candidates.extend(sorted(run_dir.glob("message_snapshot_final*.json")))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

    return None


def _snapshot_to_count_matrix(snapshot: Dict) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    by_entity = snapshot.get("by_entity", {})
    if not isinstance(by_entity, dict) or not by_entity:
        return [], [], [], np.zeros((0, 0), dtype=float)

    entity_keys = sorted(by_entity.keys())
    entity_names: List[str] = []
    all_messages = set()

    for key in entity_keys:
        info = by_entity.get(key, {})
        entity_names.append(str(info.get("entity", key)))
        for top_msg in info.get("top_messages", []):
            message = top_msg.get("message")
            if isinstance(message, str):
                all_messages.add(message)

    messages = sorted(all_messages)
    msg_to_idx = {message: idx for idx, message in enumerate(messages)}
    counts = np.zeros((len(entity_keys), len(messages)), dtype=float)

    for row_idx, key in enumerate(entity_keys):
        info = by_entity.get(key, {})
        for top_msg in info.get("top_messages", []):
            message = top_msg.get("message")
            count = top_msg.get("count", 0)
            if isinstance(message, str) and message in msg_to_idx:
                try:
                    counts[row_idx, msg_to_idx[message]] += float(count)
                except (TypeError, ValueError):
                    continue

    return entity_keys, entity_names, messages, counts


@dataclass
class ExperimentSpec:
    name: str
    relative_dir: str
    log_pattern: str = "train*"


class ExperimentAggregator:
    def __init__(self, root_dir: Path, metrics_root: Path, spec: ExperimentSpec):
        self.root_dir = root_dir.resolve()
        self.spec = spec
        self.run_dir = (self.root_dir / spec.relative_dir).resolve()
        self.output_dir = (metrics_root / "experiments" / spec.name).resolve()

        self.run_payloads: Dict[str, Dict] = {}
        self.run_snapshots: Dict[str, Dict] = {}
        self.message_reuse_rows: List[Dict[str, object]] = []
        self.final_summary: Dict[str, Dict[str, float]] = {}

    def load(self) -> None:
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.run_dir}")

        logs = _discover_logs(self.run_dir, self.spec.log_pattern)
        if not logs:
            raise FileNotFoundError(
                f"No logs found for experiment '{self.spec.name}' in {self.run_dir} "
                f"with pattern '{self.spec.log_pattern}'"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{self.spec.name}] found {len(logs)} logs")
        for log in logs:
            payload, epochs = _parse_log_file(log)
            snapshot = _load_snapshot_for_log(self.run_dir, log)
            self.run_payloads[log.name] = {
                "file": log,
                "run_label": _stable_run_label(log.name),
                "metrics": payload,
                "epochs": epochs,
            }
            if snapshot is not None:
                self.run_snapshots[log.name] = snapshot
            print(f"    - {log.name}: {len(payload['test'])} test epochs")

    def _aggregate_metric_series(self, metric_name: str, mode: str = "test") -> Tuple[List[int], List[float], List[float]]:
        all_epochs = sorted({
            epoch
            for payload in self.run_payloads.values()
            for epoch in payload["metrics"][mode].keys()
        })

        valid_epochs: List[int] = []
        means: List[float] = []
        stds: List[float] = []

        for epoch in all_epochs:
            values = []
            for payload in self.run_payloads.values():
                row = payload["metrics"][mode].get(epoch, {})
                if metric_name in row:
                    values.append(float(row[metric_name]))
            if not values:
                continue
            valid_epochs.append(epoch)
            means.append(float(np.mean(values)))
            stds.append(_sample_std(values))

        return valid_epochs, means, stds

    def _final_metric_by_run(self, metric_name: str, mode: str = "test") -> Dict[str, float]:
        out: Dict[str, float] = {}

        for run_name, payload in self.run_payloads.items():
            mode_metrics: Dict[int, Dict[str, float]] = payload["metrics"][mode]
            if not mode_metrics:
                continue
            final_epoch = max(mode_metrics.keys())
            row = mode_metrics.get(final_epoch, {})
            if metric_name in row:
                out[run_name] = float(row[metric_name])

        return out

    def _plot_loss_train_test(self) -> None:
        train_epochs, train_means, train_stds = self._aggregate_metric_series("loss", mode="train")
        test_epochs, test_means, test_stds = self._aggregate_metric_series("loss", mode="test")
        if not train_epochs and not test_epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_epochs:
            x = np.array(train_epochs)
            y = np.array(train_means)
            s = np.array(train_stds)
            ax.plot(x, y, color="#1f77b4", linewidth=2.0, label="train loss mean")
            ax.fill_between(x, y - s, y + s, color="#1f77b4", alpha=0.15)

        if test_epochs:
            x = np.array(test_epochs)
            y = np.array(test_means)
            s = np.array(test_stds)
            ax.plot(x, y, color="#ff7f0e", linewidth=2.0, label="test loss mean")
            ax.fill_between(x, y - s, y + s, color="#ff7f0e", alpha=0.15)

        ax.set_title(f"{self.spec.name}: overall loss (train/test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("loss", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_train_test.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_reward_mean_expected(self) -> None:
        me_epochs, me_means, me_stds = self._aggregate_metric_series("mean_reward", mode="test")
        ex_epochs, ex_means, ex_stds = self._aggregate_metric_series("expected_reward", mode="test")
        if not me_epochs and not ex_epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if me_epochs:
            x = np.array(me_epochs)
            y = np.array(me_means)
            s = np.array(me_stds)
            ax.plot(x, y, color="#2ca02c", linewidth=2.0, label="mean_reward (test)")
            ax.fill_between(x, y - s, y + s, color="#2ca02c", alpha=0.15)

        if ex_epochs:
            x = np.array(ex_epochs)
            y = np.array(ex_means)
            s = np.array(ex_stds)
            ax.plot(x, y, color="#9467bd", linewidth=2.0, label="expected_reward (test)")
            ax.fill_between(x, y - s, y + s, color="#9467bd", alpha=0.15)

        ax.set_title(f"{self.spec.name}: mean vs expected reward (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("reward", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.output_dir / "mean_vs_expected_reward_test.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_recon_accuracy(self) -> None:
        epochs, means, stds = self._aggregate_metric_series("recon_acc", mode="test")
        if not epochs:
            return

        x = np.array(epochs)
        y = np.array(means)
        s = np.array(stds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, color="#d62728", linewidth=2.0, label="recon_acc (test)")
        ax.fill_between(x, y - s, y + s, color="#d62728", alpha=0.15)
        ax.set_title(f"{self.spec.name}: reconstruction accuracy (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("recon_acc", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.output_dir / "recon_accuracy_test.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_topsim(self) -> None:
        epochs, means, stds = self._aggregate_metric_series("topsim", mode="test")
        if not epochs:
            return

        x = np.array(epochs)
        y = np.array(means)
        s = np.array(stds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, color="#8c564b", linewidth=2.0, label="topsim (test)")
        ax.fill_between(x, y - s, y + s, color="#8c564b", alpha=0.15)
        ax.set_title(f"{self.spec.name}: topsim (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("topsim", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.output_dir / "topsim_test.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _action_metrics(self) -> List[str]:
        preferred = [
            "hunt_rate",
            "gather_rate",
            "flee_rate",
            "rest_rate",
            "mitigate_rate",
            "endure_rate",
            "eat_rate",
            "craft_rate",
        ]

        available = set()
        for payload in self.run_payloads.values():
            for row in payload["metrics"]["test"].values():
                for key in row.keys():
                    if not key.endswith("_rate"):
                        continue
                    if key in {"valid_action_rate", "invalid_entity_idx_rate"}:
                        continue
                    available.add(key)

        ordered = [k for k in preferred if k in available]
        remaining = sorted([k for k in available if k not in ordered])
        return ordered + remaining

    def _plot_action_distribution(self) -> None:
        action_metrics = self._action_metrics()
        if not action_metrics:
            return

        fig, ax = plt.subplots(figsize=(11, 7))
        for metric in action_metrics:
            epochs, means, _stds = self._aggregate_metric_series(metric, mode="test")
            if not epochs:
                continue
            ax.plot(np.array(epochs), np.array(means), linewidth=1.8, label=metric.replace("_rate", ""))

        ax.set_title(f"{self.spec.name}: action distribution (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("rate", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "action_distribution_test.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_message_reuse_csvs(self) -> None:
        self.message_reuse_rows = []

        for run_name in sorted(self.run_snapshots.keys()):
            snapshot = self.run_snapshots[run_name]
            _entity_keys, entity_names, messages, counts = _snapshot_to_count_matrix(snapshot)
            if not entity_names or not messages or counts.size == 0:
                continue

            msg_entity_counts = (counts > 0).sum(axis=0)
            total_by_msg = counts.sum(axis=0)
            reused_indices = [i for i, n in enumerate(msg_entity_counts) if int(n) >= 2]

            run_label = _stable_run_label(run_name)
            run_id = _extract_run_id(run_label)

            detail_path = self.output_dir / f"message_reuse_{run_label}.csv"
            with open(detail_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["message", "entities_using_message", "total_count", "entity_list"])
                for i in sorted(
                    range(len(messages)),
                    key=lambda j: (int(msg_entity_counts[j]), float(total_by_msg[j])),
                    reverse=True,
                ):
                    entities = [entity_names[r] for r in range(len(entity_names)) if counts[r, i] > 0]
                    writer.writerow([
                        messages[i],
                        int(msg_entity_counts[i]),
                        float(total_by_msg[i]),
                        "|".join(entities),
                    ])

            self.message_reuse_rows.append({
                "run": run_name,
                "run_label": run_label,
                "run_id": run_id,
                "n_entities": len(entity_names),
                "n_messages": len(messages),
                "reused_messages": len(reused_indices),
                "reuse_ratio": float(len(reused_indices) / max(len(messages), 1)),
                "max_entities_on_one_message": int(np.max(msg_entity_counts)) if len(messages) > 0 else 0,
                "top_reused_message": messages[int(np.argmax(msg_entity_counts))] if len(messages) > 0 else "",
            })

        if self.message_reuse_rows:
            summary_path = self.output_dir / "message_reuse_summary.csv"
            with open(summary_path, "w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "run",
                    "run_label",
                    "run_id",
                    "n_entities",
                    "n_messages",
                    "reused_messages",
                    "reuse_ratio",
                    "max_entities_on_one_message",
                    "top_reused_message",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.message_reuse_rows)

    def _save_final_metrics_csvs(self) -> None:
        metrics = ["loss", "mean_reward", "expected_reward", "recon_acc", "topsim"]
        run_names = sorted(self.run_payloads.keys())

        final_per_run = self.output_dir / "final_metrics_per_run.csv"
        with open(final_per_run, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "run_label"] + metrics)
            for run_name in run_names:
                run_label = _stable_run_label(run_name)
                payload = self.run_payloads[run_name]
                test_metrics = payload["metrics"]["test"]
                if not test_metrics:
                    continue
                final_epoch = max(test_metrics.keys())
                row = test_metrics.get(final_epoch, {})

                values = []
                for metric in metrics:
                    value = row.get(metric)
                    values.append("" if value is None else f"{float(value):.6f}")
                writer.writerow([run_name, run_label] + values)

        summary_rows = []
        self.final_summary = {}
        for metric in metrics:
            per_run = self._final_metric_by_run(metric, mode="test")
            vals = [float(v) for v in per_run.values()]
            if not vals:
                continue
            stat = {
                "metric": metric,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": _sample_std(vals),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            summary_rows.append(stat)
            self.final_summary[metric] = stat

        final_summary = self.output_dir / "final_metrics_summary.csv"
        with open(final_summary, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "n", "mean", "std", "min", "max"])
            for row in summary_rows:
                writer.writerow([
                    row["metric"],
                    int(row["n"]),
                    f"{float(row['mean']):.6f}",
                    f"{float(row['std']):.6f}",
                    f"{float(row['min']):.6f}",
                    f"{float(row['max']):.6f}",
                ])

    def generate(self) -> None:
        self.load()
        self._plot_loss_train_test()
        self._plot_reward_mean_expected()
        self._plot_recon_accuracy()
        self._plot_topsim()
        self._plot_action_distribution()
        self._save_message_reuse_csvs()
        self._save_final_metrics_csvs()


class CrossExperimentComparator:
    def __init__(self, experiments: List[ExperimentAggregator], metrics_root: Path):
        self.experiments = experiments
        self.metrics_root = metrics_root.resolve()
        self.metrics_root.mkdir(parents=True, exist_ok=True)

    def _topsim_ranking(self) -> None:
        rows = []
        for exp in self.experiments:
            stat = exp.final_summary.get("topsim")
            if not stat:
                continue
            rows.append({
                "experiment": exp.spec.name,
                "n": int(stat["n"]),
                "mean": float(stat["mean"]),
                "std": float(stat["std"]),
            })

        if not rows:
            return

        rows.sort(key=lambda x: x["mean"], reverse=True)

        ranking_csv = self.metrics_root / "topsim_ranking_experiments.csv"
        with open(ranking_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "experiment", "topsim_mean", "topsim_std", "n_runs"])
            for idx, row in enumerate(rows, start=1):
                writer.writerow([
                    idx,
                    row["experiment"],
                    f"{row['mean']:.6f}",
                    f"{row['std']:.6f}",
                    row["n"],
                ])

        labels = [row["experiment"] for row in rows]
        means = [row["mean"] for row in rows]
        stds = [row["std"] for row in rows]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color="#1f77b4", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("final topsim mean", fontsize=11, fontweight="bold")
        ax.set_title("TopSim ranking across experiments", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(self.metrics_root / "topsim_ranking_experiments.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _topsim_trajectory(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.experiments))))
        for idx, exp in enumerate(self.experiments):
            epochs, means, stds = exp._aggregate_metric_series("topsim", mode="test")
            if not epochs:
                continue

            plotted = True
            x = np.array(epochs)
            y = np.array(means)
            s = np.array(stds)
            color = colors[idx]
            ax.plot(x, y, linewidth=2.0, color=color, label=exp.spec.name)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.12)

        if not plotted:
            plt.close()
            return

        ax.set_title("TopSim across experiments (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("topsim", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.metrics_root / "topsim_across_experiments.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _tuning_reuse_grouped_plot(self) -> None:
        tuning_experiments = [
            exp for exp in self.experiments
            if "no_recon_tuning" in str(exp.run_dir)
        ]
        if len(tuning_experiments) < 2:
            return

        run_ids = sorted({
            int(row["run_id"])
            for exp in tuning_experiments
            for row in exp.message_reuse_rows
            if row.get("run_id") is not None
        })
        if not run_ids:
            return

        x = np.arange(len(run_ids), dtype=float)
        width = 0.8 / max(1, len(tuning_experiments))
        colors = plt.cm.Set2(np.linspace(0, 1, len(tuning_experiments)))

        fig, ax = plt.subplots(figsize=(11, 6))
        for idx, exp in enumerate(tuning_experiments):
            rows_by_id = {
                int(row["run_id"]): row
                for row in exp.message_reuse_rows
                if row.get("run_id") is not None
            }
            heights = [float(rows_by_id.get(run_id, {}).get("reused_messages", 0.0)) for run_id in run_ids]
            offset = (idx - (len(tuning_experiments) - 1) / 2.0) * width
            ax.bar(x + offset, heights, width=width, color=colors[idx], label=exp.spec.name)

        ax.set_xticks(x)
        ax.set_xticklabels([f"run{run_id}" for run_id in run_ids])
        ax.set_ylabel("reused messages", fontsize=11, fontweight="bold")
        ax.set_title("Tuning message reuse grouped by run id", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.metrics_root / "tuning_message_reuse_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate(self) -> None:
        self._topsim_ranking()
        self._topsim_trajectory()
        self._tuning_reuse_grouped_plot()


def _parse_experiment_spec(raw_spec: str) -> ExperimentSpec:
    if "=" not in raw_spec:
        raise ValueError(
            f"Invalid --experiment value '{raw_spec}'. Expected: name=relative_dir[:pattern]"
        )

    name, rhs = raw_spec.split("=", 1)
    name = name.strip()
    rhs = rhs.strip()
    if not name or not rhs:
        raise ValueError(
            f"Invalid --experiment value '{raw_spec}'. Expected: name=relative_dir[:pattern]"
        )

    if ":" in rhs:
        rel_dir, pattern = rhs.split(":", 1)
        rel_dir = rel_dir.strip()
        pattern = pattern.strip() or "train*"
    else:
        rel_dir = rhs
        pattern = "train*"

    if not rel_dir:
        rel_dir = "."

    return ExperimentSpec(name=name, relative_dir=rel_dir, log_pattern=pattern)


def _default_experiment_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(name="with_recon", relative_dir="with_recon", log_pattern="train*"),
        ExperimentSpec(name="no_recon", relative_dir="no_recon", log_pattern="train*"),
        ExperimentSpec(
            name="tuning_ae03_len2",
            relative_dir="no_recon_tuning/Action_Entropy_0.3",
            log_pattern="train*",
        ),
        ExperimentSpec(
            name="tuning_ae05_len2",
            relative_dir="no_recon_tuning/Action_Entropy_0.5",
            log_pattern="train*",
        ),
        ExperimentSpec(
            name="tuning_ae05_len3",
            relative_dir="no_recon_tuning/Action_Entropy_0.5_message_3",
            log_pattern="train*",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment-level metrics for survival_game outputs"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        nargs="?",
        default="outputs",
        help="Root outputs directory containing with_recon/no_recon/no_recon_tuning",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Optional override experiment spec: name=relative_dir[:pattern] (repeatable)",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    metrics_root = (root_dir / "metrics").resolve()

    specs = [_parse_experiment_spec(raw) for raw in args.experiment] if args.experiment else _default_experiment_specs()

    print("=" * 70)
    print("SURVIVAL GAME - EXPERIMENT METRICS")
    print("=" * 70)
    print(f"Root outputs: {root_dir}")
    print(f"Metrics root: {metrics_root}")
    print(f"Experiments: {len(specs)}")

    experiments: List[ExperimentAggregator] = []
    for spec in specs:
        print(f"\n[Experiment: {spec.name}]")
        aggregator = ExperimentAggregator(root_dir=root_dir, metrics_root=metrics_root, spec=spec)
        aggregator.generate()
        experiments.append(aggregator)

    print("\n[Cross-experiment comparisons]")
    comparator = CrossExperimentComparator(experiments=experiments, metrics_root=metrics_root)
    comparator.generate()

    print("\n" + "=" * 70)
    print("Process completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
