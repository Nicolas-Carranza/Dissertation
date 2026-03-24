#!/usr/bin/env python3
"""Generate dissertation tables for run11 vs run12.

Outputs (by default into ../outputs):
- run11_run12_core_table.csv
- run11_run12_language_table.csv
- run11_run12_tables.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


def defaults() -> Dict[str, Path]:
    dissertation_root = Path(__file__).resolve().parents[2]
    out_dir = dissertation_root / "outputs"
    return {
        "out_dir": out_dir,
        "run11_log": out_dir / "train_run11.log",
        "run12_log": out_dir / "train_run12.log",
        "run11_progression": out_dir / "message_progression_run11.jsonl",
        "run12_progression": out_dir / "message_progression_run12.jsonl",
    }


def parse_args() -> argparse.Namespace:
    d = defaults()
    parser = argparse.ArgumentParser(description="Generate run11 vs run12 summary tables")
    parser.add_argument("--run11_log", type=Path, default=d["run11_log"])
    parser.add_argument("--run12_log", type=Path, default=d["run12_log"])
    parser.add_argument("--run11_progression", type=Path, default=d["run11_progression"])
    parser.add_argument("--run12_progression", type=Path, default=d["run12_progression"])
    parser.add_argument("--out_dir", type=Path, default=d["out_dir"])
    return parser.parse_args()


def validate_paths(paths: List[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def load_progression(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: int(r["epoch"]))
    return records


def parse_log(path: Path) -> Dict[str, object]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    survival_rates = [float(x) for x in re.findall(r"Survival rate:\s*([0-9]+(?:\.[0-9]+)?)%", content)]

    last_test_recon_acc: Optional[float] = None
    entropy_train: List[float] = []
    entropy_test: List[float] = []
    posdis_test: List[float] = []
    bosdis_test_count = 0
    topsim_test_count = 0

    for raw in lines:
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        mode = obj.get("mode")

        if mode == "test" and "recon_acc" in obj:
            last_test_recon_acc = float(obj["recon_acc"])

        if "entropy" in obj:
            if mode == "train":
                entropy_train.append(float(obj["entropy"]))
            elif mode == "test":
                entropy_test.append(float(obj["entropy"]))

        if mode == "test" and "posdis" in obj:
            if obj["posdis"] is not None:
                posdis_test.append(float(obj["posdis"]))

        if mode == "test" and "bosdis" in obj:
            bosdis_test_count += 1

        if mode == "test" and "topsim" in obj:
            topsim_test_count += 1

    return {
        "survival_rates": survival_rates,
        "last_test_recon_acc": last_test_recon_acc,
        "entropy_train": entropy_train,
        "entropy_test": entropy_test,
        "posdis_test": posdis_test,
        "bosdis_test_count": bosdis_test_count,
        "topsim_test_count": topsim_test_count,
    }


def convergence_label(prog: List[dict]) -> str:
    totals = [int(r["total_unique_messages"]) for r in prog]
    epochs = [int(r["epoch"]) for r in prog]

    first_40_epoch = None
    for e, t in zip(epochs, totals):
        if t == 40:
            first_40_epoch = e
            break

    if first_40_epoch is None:
        return "No (never reached 40)"
    if totals[-1] < 40:
        return "No (unstable tail)"

    tail = totals[-10:] if len(totals) >= 10 else totals
    if tail and all(t == 40 for t in tail) and first_40_epoch <= 100:
        return "Yes (stable from early on)"
    if tail and all(t == 40 for t in tail):
        return "Yes (stable late)"
    return "Partial (reached 40 but unstable)"


def percent(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}%"


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def core_rows(prog: List[dict], log_info: Dict[str, object]) -> Dict[str, str]:
    totals = [int(r["total_unique_messages"]) for r in prog]
    survival_rates = log_info["survival_rates"]
    recon = log_info["last_test_recon_acc"]

    mean_survival = mean(survival_rates) if survival_rates else 0.0
    final_survival = survival_rates[-1] if survival_rates else 0.0

    return {
        "Final unique messages": str(totals[-1]),
        "Max unique during training": str(max(totals) if totals else 0),
        "Final recon_acc": fmt(recon, 4),
        "Mean survival (60 evals)": percent(mean_survival, 2),
        "Final eval survival": percent(final_survival, 0),
        "Convergence of message inventory": convergence_label(prog),
    }


def language_rows(log_info: Dict[str, object]) -> Dict[str, str]:
    ent_train = log_info["entropy_train"]
    ent_test = log_info["entropy_test"]
    pos_test = log_info["posdis_test"]

    return {
        "Entropy entries (train + test)": str(len(ent_train) + len(ent_test)),
        "Final test entropy": fmt(ent_test[-1] if ent_test else None, 4),
        "Mean test entropy": fmt(mean(ent_test) if ent_test else None, 4),
        "Disent posdis entries (test)": str(len(pos_test)),
        "Final test posdis": fmt(pos_test[-1] if pos_test else None, 4),
        "Mean test posdis": fmt(mean(pos_test) if pos_test else None, 4),
        "Disent bosdis entries (test)": str(log_info["bosdis_test_count"]),
        "TopSim entries (test)": str(log_info["topsim_test_count"]),
    }


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def markdown_table(header: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    validate_paths([
        args.run11_log,
        args.run12_log,
        args.run11_progression,
        args.run12_progression,
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prog11 = load_progression(args.run11_progression)
    prog12 = load_progression(args.run12_progression)
    log11 = parse_log(args.run11_log)
    log12 = parse_log(args.run12_log)

    core11 = core_rows(prog11, log11)
    core12 = core_rows(prog12, log12)
    lang11 = language_rows(log11)
    lang12 = language_rows(log12)

    core_order = [
        "Final unique messages",
        "Max unique during training",
        "Final recon_acc",
        "Mean survival (60 evals)",
        "Final eval survival",
        "Convergence of message inventory",
    ]
    lang_order = [
        "Entropy entries (train + test)",
        "Final test entropy",
        "Mean test entropy",
        "Disent posdis entries (test)",
        "Final test posdis",
        "Mean test posdis",
        "Disent bosdis entries (test)",
        "TopSim entries (test)",
    ]

    core_data = [[k, core11[k], core12[k]] for k in core_order]
    lang_data = [[k, lang11[k], lang12[k]] for k in lang_order]

    core_csv = args.out_dir / "run11_run12_core_table.csv"
    lang_csv = args.out_dir / "run11_run12_language_table.csv"
    md_path = args.out_dir / "run11_run12_tables.md"

    write_csv(core_csv, ["Metric", "Run 11 (decay 0.99)", "Run 12 (no decay, 1.0)"], core_data)
    write_csv(lang_csv, ["Language Metric", "Run 11 (decay 0.99)", "Run 12 (no decay, 1.0)"], lang_data)

    core_md = markdown_table(["Metric", "Run 11 (decay 0.99)", "Run 12 (no decay, 1.0)"], core_data)
    lang_md = markdown_table(["Language Metric", "Run 11 (decay 0.99)", "Run 12 (no decay, 1.0)"], lang_data)

    md_content = (
        "# Run 11 vs Run 12 Tables\n\n"
        "## Core Comparison\n\n"
        f"{core_md}\n\n"
        "## Language Analysis Metrics\n\n"
        f"{lang_md}\n"
    )
    md_path.write_text(md_content, encoding="utf-8")

    print("Wrote:")
    print(f"  - {core_csv}")
    print(f"  - {lang_csv}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
