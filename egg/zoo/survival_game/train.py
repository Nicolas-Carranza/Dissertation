#!/usr/bin/env python3
"""
Survival Game — train.py
==========================

Main entry point for training the Survival Game with emergent communication.

Supports two training modes:
  --mode rf   Reinforce (policy gradient, default)
  --mode gs   Gumbel-Softmax (differentiable relaxation)

Usage:
    python -m egg.zoo.survival_game.train --mode gs [OPTIONS]

Example (Gumbel-Softmax):
    python -m egg.zoo.survival_game.train \\
        --mode gs \\
        --n_epochs 200 \\
        --batch_size 64 \\
        --vocab_size 50 \\
        --max_len 2 \\
        --temperature 1.0 \\
        --sender_hidden 128 \\
        --receiver_hidden 128 \\
        --lr 1e-3 \\
        --n_episodes 10000 \\
        --eval_freq 5

Example (Reinforce):
    python -m egg.zoo.survival_game.train \\
        --mode rf \\
        --n_epochs 50 \\
        --batch_size 64 \\
        --sender_entropy_coeff 0.1 \\
        --lr 1e-3

For a full list of options:
    python -m egg.zoo.survival_game.train --help
"""

import argparse
import io
import os
import sys
from typing import List, Optional

import torch
import egg.core as core

from egg.core.language_analysis import Disent, MessageEntropy, TopographicSimilarity
from egg.zoo.survival_game.callbacks import (
    MessageAnalyzer,
    SurvivalGameEvaluator,
)
from egg.zoo.survival_game.data import get_dataloaders
from egg.zoo.survival_game.games import build_game


def get_params(params: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    First adds game-specific flags, then passes to core.init() which
    adds the standard EGG parameters (batch_size, vocab_size, max_len, lr,
    n_epochs, etc.) and handles seeding / CUDA setup.
    """
    parser = argparse.ArgumentParser(
        description="Survival Game — emergent communication experiment"
    )

    # ---- Agent architecture ----
    parser.add_argument(
        "--mode", type=str, default="rf",
        choices=["rf", "gs"],
        help="Training mode: 'rf' (Reinforce) or 'gs' (Gumbel-Softmax) (default: rf)",
    )
    parser.add_argument(
        "--sender_hidden", type=int, default=128,
        help="Hidden size for Sender MLP + RNN (default: 128)",
    )
    parser.add_argument(
        "--receiver_hidden", type=int, default=128,
        help="Hidden size for Receiver MLP + RNN (default: 128)",
    )
    parser.add_argument(
        "--sender_embedding", type=int, default=32,
        help="Embedding dim for Sender RNN input (default: 32)",
    )
    parser.add_argument(
        "--receiver_embedding", type=int, default=32,
        help="Embedding dim for Receiver RNN input (default: 32)",
    )
    parser.add_argument(
        "--sender_cell", type=str, default="lstm",
        choices=["rnn", "gru", "lstm"],
        help="RNN cell type for Sender (default: lstm)",
    )
    parser.add_argument(
        "--receiver_cell", type=str, default="lstm",
        choices=["rnn", "gru", "lstm"],
        help="RNN cell type for Receiver (default: lstm)",
    )

    # ---- Reinforce ----
    parser.add_argument(
        "--sender_entropy_coeff", type=float, default=0.1,
        help="Entropy regularisation coefficient for Sender, RF mode only (default: 0.1)",
    )
    parser.add_argument(
        "--receiver_entropy_coeff", type=float, default=0.05,
        help="Entropy regularisation coefficient for Receiver, RF mode only (default: 0.05)",
    )
    parser.add_argument(
        "--reward_scale", type=float, default=0.2,
        help="Scaling factor for reward → loss conversion (default: 0.2)",
    )
    parser.add_argument(
        "--recon_weight", type=float, default=2.0,
        help="Weight of auxiliary entity-reconstruction loss, GS mode only (default: 2.0)",
    )
    parser.add_argument(
        "--action_entropy_coeff", type=float, default=0.1,
        help="Entropy bonus for action distribution, GS mode only (default: 0.1)",
    )
    parser.add_argument(
        "--action_temperature", type=float, default=2.0,
        help="Temperature for action softmax (higher=more exploration), GS mode (default: 2.0)",
    )
    parser.add_argument(
        "--reward_normalise", action="store_true",
        help="Normalise per-sample reward matrix to mean=0, std=1 (default: off)",
    )

    # ---- Gumbel-Softmax ----
    parser.add_argument(
        "--temperature", type=float, default=2.0,
        help="Initial GS temperature for Sender, GS mode only (default: 2.0)",
    )
    parser.add_argument(
        "--temperature_decay", type=float, default=0.9,
        help="Temperature decay per epoch, GS mode only (default: 0.9)",
    )
    parser.add_argument(
        "--temperature_minimum", type=float, default=0.1,
        help="Minimum temperature, GS mode only (default: 0.1)",
    )

    # ---- Data generation ----
    parser.add_argument(
        "--n_episodes", type=int, default=10000,
        help="Total episodes to generate (split into train/val/test) (default: 10000)",
    )
    parser.add_argument(
        "--max_turns", type=int, default=20,
        help="Turns per simulated episode (default: 20)",
    )
    parser.add_argument(
        "--data_seed", type=int, default=42,
        help="Seed for data generation reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8,
        help="Fraction of episodes for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of episodes for validation (default: 0.1)",
    )

    # ---- Evaluation ----
    parser.add_argument(
        "--eval_freq", type=int, default=5,
        help="Run full-episode evaluation every N epochs (0 = never, default: 5)",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=100,
        help="Number of episodes per evaluation (default: 100)",
    )
    parser.add_argument(
        "--analyze_freq", type=int, default=10,
        help="Analyze messages every N epochs (0 = never, default: 10)",
    )
    parser.add_argument(
        "--top_k_messages", type=int, default=10,
        help="Top-K messages to log per entity per snapshot (default: 10)",
    )
    parser.add_argument(
        "--include_all_messages", action="store_true",
        help="Log full per-entity message histograms in snapshots (can be large)",
    )
    parser.add_argument(
        "--message_snapshot_max_mb", type=int, default=0,
        help="Max snapshot JSONL size in MB before append stops (0 = unlimited)",
    )
    parser.add_argument(
        "--run_name", type=str, default="",
        help="Optional label used in output filenames (example: run9)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to store run artifacts (logs/snapshots) (default: outputs)",
    )
    parser.add_argument(
        "--log_file", type=str, default="",
        help="If set, tee full stdout/stderr stream to this file",
    )
    parser.add_argument(
        "--message_progression_file", type=str, default="",
        help="If set, append per-analysis message snapshots as JSONL",
    )
    parser.add_argument(
        "--final_snapshot_file", type=str, default="",
        help="If set, write final message snapshot JSON at last epoch",
    )
    parser.add_argument(
        "--track_message_entropy", action="store_true",
        help="Enable MessageEntropy callback from egg.core.language_analysis",
    )
    parser.add_argument(
        "--track_topsim", action="store_true",
        help="Enable TopographicSimilarity callback from egg.core.language_analysis",
    )
    parser.add_argument(
        "--topsim_max_samples", type=int, default=2000,
        help="Max samples used by TopSim per split (0 = full set; default: 2000)",
    )
    parser.add_argument(
        "--track_disent", action="store_true",
        help="Enable Disent callback from egg.core.language_analysis",
    )

    # core.init() will add standard EGG params and parse
    opts = core.init(arg_parser=parser, params=params)
    return opts


def main(params: Optional[List[str]] = None):
    """Build all components and start training."""
    opts = get_params(params)

    # Resolve output paths once so all callbacks share a consistent naming scheme.
    out_dir = opts.output_dir
    run_suffix = f"_{opts.run_name}" if opts.run_name else ""
    default_log_file = os.path.join(out_dir, f"train{run_suffix}.log")
    default_progression_file = os.path.join(out_dir, f"message_progression{run_suffix}.jsonl")
    default_final_snapshot_file = os.path.join(out_dir, f"message_snapshot_final{run_suffix}.json")

    log_file = opts.log_file or default_log_file
    message_progression_file = opts.message_progression_file or default_progression_file
    final_snapshot_file = opts.final_snapshot_file or default_final_snapshot_file

    _setup_stdio_tee(log_file)

    # ---- Print configuration ----
    print("=" * 60)
    print("  Survival Game — Training Configuration")
    print("=" * 60)
    print(f"  Mode:               {opts.mode.upper()}"
          f" ({'Gumbel-Softmax' if opts.mode == 'gs' else 'Reinforce'})")
    print(f"  Sender hidden:      {opts.sender_hidden}")
    print(f"  Receiver hidden:    {opts.receiver_hidden}")
    print(f"  Sender embedding:   {opts.sender_embedding}")
    print(f"  Receiver embedding: {opts.receiver_embedding}")
    print(f"  Sender cell:        {opts.sender_cell}")
    print(f"  Receiver cell:      {opts.receiver_cell}")
    print(f"  Vocab size:         {opts.vocab_size}")
    print(f"  Max message len:    {opts.max_len}")
    if opts.mode == "gs":
        print(f"  Temperature:        {opts.temperature}")
        print(f"  Temp decay:         {opts.temperature_decay}")
        print(f"  Temp minimum:       {opts.temperature_minimum}")
        print(f"  Recon weight:       {opts.recon_weight}")
        print(f"  Action entropy:     {opts.action_entropy_coeff}")
        print(f"  Action temperature: {opts.action_temperature}")
        print(f"  Reward normalise:   {opts.reward_normalise}")
    else:
        print(f"  Sender entropy:     {opts.sender_entropy_coeff}")
        print(f"  Receiver entropy:   {opts.receiver_entropy_coeff}")
    print(f"  Reward scale:       {opts.reward_scale}")
    print(f"  Learning rate:      {opts.lr}")
    print(f"  Batch size:         {opts.batch_size}")
    print(f"  Epochs:             {opts.n_epochs}")
    print(f"  Total episodes:     {opts.n_episodes}")
    print(f"  Split:              {opts.train_frac:.0%} / {opts.val_frac:.0%} / {1-opts.train_frac-opts.val_frac:.0%} (train/val/test)")
    print(f"  Max turns/episode:  {opts.max_turns}")
    print(f"  Eval freq:          {opts.eval_freq}")
    print(f"  Output dir:         {out_dir}")
    print(f"  Log file:           {log_file}")
    print(f"  Message progression:{message_progression_file}")
    print(f"  Final snapshot:     {final_snapshot_file}")
    print(f"  Device:             {opts.device}")
    print("=" * 60)
    print()

    # ---- Data ----
    print("Generating data...")
    train_loader, val_loader, test_loader = get_dataloaders(opts)

    # ---- Game ----
    print("Building game...")
    game = build_game(opts)

    # ---- Optimizer ----
    optimizer = core.build_optimizer(game.parameters())

    # ---- Callbacks ----
    callbacks = [
        core.ConsoleLogger(print_train_loss=True, as_json=True),
    ]

    # GS temperature annealing
    if opts.mode == "gs":
        callbacks.append(
            core.TemperatureUpdater(
                agent=game.sender,
                decay=opts.temperature_decay,
                minimum=opts.temperature_minimum,
            )
        )

    # Survival game evaluator (full episode rollouts)
    if opts.eval_freq > 0:
        callbacks.append(
            SurvivalGameEvaluator(
                n_episodes=opts.eval_episodes,
                max_turns=opts.max_turns,
                eval_freq=opts.eval_freq,
            )
        )

    # Message analyzer
    if opts.analyze_freq > 0:
        callbacks.append(
            MessageAnalyzer(
                analyze_freq=opts.analyze_freq,
                progression_path=message_progression_file,
                final_snapshot_path=final_snapshot_file,
                total_epochs=opts.n_epochs,
                top_k_messages=opts.top_k_messages,
                include_all_messages=opts.include_all_messages,
                max_snapshot_bytes=opts.message_snapshot_max_mb * 1_000_000,
            )
        )

    # Optional language-analysis callbacks from egg.core.
    if opts.track_message_entropy:
        callbacks.append(MessageEntropy(print_train=True, is_gumbel=(opts.mode == "gs")))

    if opts.track_topsim:
        callbacks.append(
            TopographicSimilarity(
                compute_topsim_train_set=False,
                compute_topsim_test_set=True,
                is_gumbel=(opts.mode == "gs"),
                max_samples=opts.topsim_max_samples,
            )
        )

    if opts.track_disent:
        callbacks.append(
            Disent(
                is_gumbel=(opts.mode == "gs"),
                compute_posdis=True,
                compute_bosdis=True,
                vocab_size=opts.vocab_size,
                print_train=False,
                print_test=True,
            )
        )

    # Checkpoint saving
    if opts.checkpoint_dir:
        callbacks.append(
            core.CheckpointSaver(
                checkpoint_path=opts.checkpoint_dir,
                checkpoint_freq=opts.checkpoint_freq,
            )
        )

    # Tensorboard
    if opts.tensorboard:
        callbacks.append(core.TensorboardLogger())

    # ---- Trainer ----
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )

    # ---- Train! ----
    print("Starting training...")
    trainer.train(n_epochs=opts.n_epochs)

    # ---- Final test-set evaluation ----
    print("\n" + "=" * 60)
    print("  Final Test Set Evaluation")
    print("=" * 60)
    game.eval()
    device = next(game.parameters()).device
    test_loss = 0.0
    test_n = 0
    with torch.no_grad():
        for batch in test_loader:
            sender_input, labels, receiver_input = batch
            sender_input = sender_input.to(device)
            labels = labels.to(device)
            receiver_input = receiver_input.to(device)
            # Forward pass through the full game
            # EGG games return (loss, interaction) regardless of mode
            loss_val, interaction = game(
                sender_input, labels, receiver_input,
            )
            test_loss += loss_val.sum().item()
            test_n += sender_input.size(0)
    if test_n > 0:
        test_loss /= test_n
        # Extract metrics from the last batch's interaction for reporting
        aux = interaction.aux if interaction.aux else {}
        print(f"  Test samples:    {test_n}")
        print(f"  Test loss:       {test_loss:.4f}")
        for key in ["recon_acc", "mean_reward", "expected_reward"]:
            if key in aux:
                val = aux[key].mean().item() if aux[key].dim() > 0 else aux[key].item()
                print(f"  Test {key}: {val:.4f}")
    print("=" * 60)
    game.train()

    print("\nTraining complete.")
    core.close()


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _setup_stdio_tee(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_stream = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_stream)
    sys.stderr = _Tee(sys.stderr, log_stream)


if __name__ == "__main__":
    main(sys.argv[1:])
