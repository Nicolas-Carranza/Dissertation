#!/usr/bin/env python3
"""
Run multiple MNIST EGG experiments with reproducible seed and non-overwriting outputs.

For each run, this script will:
1) train with play_mnist.py and save checkpoint + train log
2) generate compact message analysis (top-k only by default)
3) generate confusion matrices
4) generate a loss-only plot from the train log

Example:
python run_mnist_experiments.py \
  --seed 42 \
  --sender-entropy-coeffs 0.001,0.01 \
  --n-epochs 10 --batch-size 128 --lr 0.001 \
  --max-len 10 --vocab-size 50 --n-distractors 2

  python run_mnist_experiments.py --seed 42 --sender-entropy-coeffs 0.001,0.003,0.01 --n-epochs 10 --batch-size 128 --lr 0.001 --max-len 10 --vocab-size 50 --n-distractors 2 --samples-per-digit 1200 --top-k 5
"""

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="rf", choices=["rf"], help="Training mode for this runner (RF only)")
    p.add_argument("--sender-entropy-coeffs", type=str, default="0.001")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-len", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=50)
    p.add_argument("--n-distractors", type=int, default=2)
    p.add_argument("--samples-per-digit", type=int, default=1200)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-root", type=str, default="./runs")
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args()


def run_and_tee(cmd, cwd: Path, log_path: Path):
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(shlex.quote(x) for x in cmd)}")


def run_simple(cmd, cwd: Path, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sweep_values = [c.strip() for c in args.sender_entropy_coeffs.split(",") if c.strip()]
    if not sweep_values:
        raise ValueError("No sender entropy coefficients provided.")

    for idx, sweep_value in enumerate(sweep_values, start=1):
        run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mnist_run_{run_stamp}_{idx:02d}_mode-{args.mode}_ent-{sweep_value}_seed-{args.seed}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        checkpoint_path = run_dir / "checkpoint.pth"
        train_log_path = run_dir / "train.log"
        analysis_path = run_dir / "message_summary.json"
        confusion_prefix = run_dir / "confusion"
        loss_png = run_dir / "loss.png"

        train_cmd = [
            sys.executable,
            "play_mnist.py",
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--n_epochs",
            str(args.n_epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--max_len",
            str(args.max_len),
            "--vocab_size",
            str(args.vocab_size),
            "--n_distractors",
            str(args.n_distractors),
            "--save_checkpoint",
            str(checkpoint_path),
        ]

        train_cmd.extend(["--sender_entropy_coeff", str(sweep_value)])

        analyze_cmd = [
            sys.executable,
            "analyze_messages.py",
            "--checkpoint",
            str(checkpoint_path),
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--samples_per_digit",
            str(args.samples_per_digit),
            "--vocab_size",
            str(args.vocab_size),
            "--max_len",
            str(args.max_len),
            "--n_distractors",
            str(args.n_distractors),
            "--top_k",
            str(args.top_k),
            "--summary_only",
            "--output",
            str(analysis_path),
        ]

        confusion_cmd = [
            sys.executable,
            "generate_confusion.py",
            "--checkpoint",
            str(checkpoint_path),
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--samples_per_digit",
            str(args.samples_per_digit),
            "--vocab_size",
            str(args.vocab_size),
            "--max_len",
            str(args.max_len),
            "--n_distractors",
            str(args.n_distractors),
            "--output_prefix",
            str(confusion_prefix),
        ]

        loss_cmd = [
            sys.executable,
            "plot_loss_from_log.py",
            "--log",
            str(train_log_path),
            "--output",
            str(loss_png),
            "--title",
            f"Loss curve ({run_name})",
        ]

        print("=" * 80)
        print(f"Running experiment: {run_name}")
        print(f"Output directory: {run_dir}")
        print("=" * 80)

        if args.dry_run:
            print("DRY RUN: would execute")
            for cmd in [train_cmd, analyze_cmd, confusion_cmd, loss_cmd]:
                print("$", " ".join(shlex.quote(x) for x in cmd))
            continue

        run_and_tee(train_cmd, cwd=script_dir, log_path=train_log_path)
        run_simple(analyze_cmd, cwd=script_dir)
        run_simple(confusion_cmd, cwd=script_dir)
        run_simple(loss_cmd, cwd=script_dir)

        print(f"Completed run: {run_name}")
        print(f"- checkpoint: {checkpoint_path}")
        print(f"- train log: {train_log_path}")
        print(f"- message summary: {analysis_path}")
        print(f"- confusion: {confusion_prefix}_counts.png and {confusion_prefix}_accuracy.png")
        print(f"- loss plot: {loss_png}")


if __name__ == "__main__":
    main()
