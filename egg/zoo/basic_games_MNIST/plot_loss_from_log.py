#!/usr/bin/env python3
"""
Plot train/test loss over epochs from a JSON-lines training log.

Expected log format: lines containing JSON objects with keys including:
  - mode: "train" or "test"
  - epoch: integer epoch id
  - loss: scalar loss value

Usage:
  python plot_loss_from_log.py --log ./runs/exp_01/train.log --output ./runs/exp_01/loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to train log file")
    parser.add_argument("--output", required=True, help="Path to output PNG")
    parser.add_argument("--title", default="Loss Over Epochs", help="Plot title")
    return parser.parse_args()


def load_loss_series(log_path: Path):
    train = {}
    test = {}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = rec.get("epoch")
            mode = rec.get("mode")
            loss = rec.get("loss")
            if epoch is None or mode not in {"train", "test"} or loss is None:
                continue

            if mode == "train":
                train[int(epoch)] = float(loss)
            else:
                test[int(epoch)] = float(loss)

    train_epochs = sorted(train)
    test_epochs = sorted(test)
    return train_epochs, [train[e] for e in train_epochs], test_epochs, [test[e] for e in test_epochs]


def main():
    args = parse_args()
    log_path = Path(args.log).resolve()
    out_path = Path(args.output).resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr_e, tr_v, te_e, te_v = load_loss_series(log_path)

    if not tr_e and not te_e:
        raise RuntimeError("No train/test loss entries found in log.")

    plt.figure(figsize=(9, 5.5))
    if tr_e:
        plt.plot(tr_e, tr_v, "o-", label="Train loss", linewidth=2)
    if te_e:
        plt.plot(te_e, te_v, "s-", label="Test loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved loss plot: {out_path}")


if __name__ == "__main__":
    main()
