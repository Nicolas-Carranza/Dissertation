#!/usr/bin/env python3
"""
Survival Game — generate_metrics.py
====================================

Generates comprehensive visualization of training metrics from any run output.

Reads train_run.log (JSON lines format) and generates plots for:
    - Loss (train vs test) over epochs
    - Recon accuracy over epochs
    - Mean reward over epochs
    - Recon loss over epochs
    - Action distribution rates over epochs
    - TopSim over epochs
    - PosDis over epochs
    - BosDis over epochs
    - Message entropy over epochs

Saves all plots to a 'metrics' folder within the run directory.

Usage:
    python generate_metrics.py <path_to_run_directory>
    python generate_metrics.py outputs/  # if run in egg_repo directory

Example:
    python generate_metrics.py /path/to/egg_repo/outputs/
"""

import json
import argparse
import sys
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def _extract_run_id(name: str) -> Optional[str]:
    """Extract numeric run id from a filename stem like train_run19."""
    match = re.search(r"run(\d+)", name)
    if not match:
        return None
    return match.group(1)


def _sanitize_stem_for_filename(stem: str) -> str:
    """Convert arbitrary stem into a safe filename token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    token = re.sub(r"_+", "_", token).strip("._")
    return token or "run"


def _stable_run_label(file_name: str) -> str:
    """Build stable output label while preserving legacy runNN names when possible."""
    stem = Path(file_name).stem
    run_id = _extract_run_id(stem)
    if run_id is not None and re.fullmatch(r"train_run\d+", stem):
        return f"run{run_id}"
    return _sanitize_stem_for_filename(stem)


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    """Keep insertion order while removing duplicates."""
    seen = set()
    out: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _derive_log_suffixes(log_stem: str) -> List[str]:
    """Derive candidate suffixes used by train.py naming contract."""
    suffixes: List[str] = []

    def push(value: str) -> None:
        if value not in suffixes:
            suffixes.append(value)

    push(log_stem)
    if log_stem.startswith("train_"):
        push(log_stem[len("train_"):])
    if log_stem.startswith("train"):
        push(log_stem[len("train"):].lstrip("_"))
    return suffixes


def _resolve_artifact_for_log(
    run_dir: Path,
    log_stem: str,
    artifact_prefix: str,
    extension: str,
) -> Optional[Path]:
    """Resolve artifact path for a log stem while avoiding ambiguous fallbacks."""
    suffixes = _derive_log_suffixes(log_stem)
    run_id = _extract_run_id(log_stem)

    exact_candidates: List[Path] = []
    for suffix in suffixes:
        if suffix:
            exact_candidates.append(run_dir / f"{artifact_prefix}_{suffix}{extension}")
            if suffix.startswith("_"):
                exact_candidates.append(run_dir / f"{artifact_prefix}{suffix}{extension}")
        else:
            exact_candidates.append(run_dir / f"{artifact_prefix}{extension}")

    if run_id is not None:
        exact_candidates.append(run_dir / f"{artifact_prefix}_run{run_id}{extension}")

    for path in _dedupe_paths(exact_candidates):
        if path.exists():
            return path

    fuzzy_candidates: List[Path] = []
    for suffix in suffixes:
        if suffix:
            fuzzy_candidates.extend(sorted(run_dir.glob(f"{artifact_prefix}*{suffix}*{extension}")))
    if run_id is not None:
        fuzzy_candidates.extend(sorted(run_dir.glob(f"{artifact_prefix}*run{run_id}*{extension}")))
    fuzzy_candidates = _dedupe_paths(fuzzy_candidates)

    if len(fuzzy_candidates) == 1:
        return fuzzy_candidates[0]
    if len(fuzzy_candidates) > 1:
        scored = []
        for candidate in fuzzy_candidates:
            name = candidate.stem.lower()
            score = 0
            for suffix in suffixes:
                if suffix and suffix.lower() in name:
                    score = max(score, len(suffix))
            if run_id is not None and f"run{run_id}" in name:
                score += 5
            scored.append((score, len(candidate.name), candidate))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if len(scored) >= 2 and scored[0][0] > scored[1][0]:
            return scored[0][2]
        return None

    all_candidates = sorted(run_dir.glob(f"{artifact_prefix}*{extension}"))
    if len(all_candidates) == 1:
        return all_candidates[0]
    return None


class MetricsGenerator:
    """
    Generates metric visualizations from training log files.
    """
    
    def __init__(
        self,
        run_dir: Path,
        log_file: Optional[str] = None,
        hungarian_mode: str = "both",
        hungarian_epochs: Optional[str] = None,
    ):
        """
        Initialize metrics generator.
        
        Args:
            run_dir: Path to the run directory containing train_run.log
            log_file: Optional explicit log filename
            hungarian_mode: Hungarian objective mode: frequency, distance, both
            hungarian_epochs: Comma-separated epochs, e.g. "10,20,30,40,50"
        """
        self.run_dir = Path(run_dir).resolve()
        self.log_file = self._find_log_file(log_file)
        self.metrics_dir = self.run_dir / "metrics"
        self.hungarian_mode = hungarian_mode
        self.hungarian_epochs = self._parse_hungarian_epochs(hungarian_epochs)
        self.hungarian_results: Dict[str, Dict] = {}
        
        # Sanity checks
        self._validate_input()
        self._create_output_dir()
        
        # Data storage
        self.metrics = {
            'train': {},
            'test': {}
        }
        self.epochs = set()

    @staticmethod
    def _parse_hungarian_epochs(raw_epochs: Optional[str]) -> List[int]:
        """Parse comma-separated Hungarian analysis epochs."""
        if not raw_epochs:
            return [10, 20, 30, 40, 50]

        parsed: List[int] = []
        for tok in raw_epochs.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                parsed.append(int(tok))
            except ValueError:
                pass
        return sorted(set(parsed)) if parsed else [10, 20, 30, 40, 50]

    def _find_log_file(self, explicit_log_file: Optional[str]) -> Path:
        """Find the training log file with sane fallbacks."""
        if explicit_log_file:
            explicit_path = Path(explicit_log_file)
            if not explicit_path.is_absolute():
                explicit_path = self.run_dir / explicit_path
            return explicit_path.resolve()

        # Common default first
        default_log = self.run_dir / "train_run.log"
        if default_log.exists():
            return default_log

        default_txt = self.run_dir / "train_run.txt"
        if default_txt.exists():
            return default_txt

        # train.py default when run_name is not set.
        default_train_log = self.run_dir / "train.log"
        if default_train_log.exists():
            return default_train_log

        default_train_txt = self.run_dir / "train.txt"
        if default_train_txt.exists():
            return default_train_txt

        # Any run-specific log, e.g. train_run13.log
        candidates = sorted(self.run_dir.glob("train_run*.log"))
        if candidates:
            return candidates[0]

        txt_candidates = sorted(self.run_dir.glob("train_run*.txt"))
        if txt_candidates:
            return txt_candidates[0]

        generic_candidates = sorted(self.run_dir.glob("train*.log"))
        if generic_candidates:
            return generic_candidates[0]

        generic_txt_candidates = sorted(self.run_dir.glob("train*.txt"))
        if generic_txt_candidates:
            return generic_txt_candidates[0]

        # Let validation raise a clear error
        return default_log
        
    def _validate_input(self) -> None:
        """Validate that run directory and log file exist."""
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        
        if not self.log_file.exists():
            available = sorted(self.run_dir.glob("*.log")) + sorted(self.run_dir.glob("*.txt"))
            available_names = ", ".join(p.name for p in available) if available else "none"
            raise FileNotFoundError(
                f"Train log not found: {self.log_file}. Available log/txt files: {available_names}"
            )
        
        print(f"✓ Run directory found: {self.run_dir}")
        print(f"✓ Log file found: {self.log_file}")
    
    def _create_output_dir(self) -> None:
        """Create metrics output directory if it doesn't exist."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Metrics directory ready: {self.metrics_dir}")
    
    def load_metrics(self) -> None:
        """Parse the training log and extract metrics."""
        print("\n[Loading metrics...]")
        
        train_metrics = {}
        test_metrics = {}
        skipped_non_json = 0
        skipped_invalid_json = 0
        
        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_stripped = line.strip()

                # Ignore banner/progress lines quietly
                if not line_stripped or not line_stripped.startswith('{'):
                    skipped_non_json += 1
                    continue

                try:
                    data = json.loads(line_stripped)
                    if not data:  # skip empty lines
                        continue
                    
                    epoch = data.get('epoch')
                    mode = data.get('mode')
                    
                    if epoch is None or mode is None:
                        continue
                    
                    self.epochs.add(epoch)
                    
                    # Initialize epoch dict if needed
                    if epoch not in train_metrics:
                        train_metrics[epoch] = {}
                    if epoch not in test_metrics:
                        test_metrics[epoch] = {}
                    
                    # Store metrics
                    metrics_dict = train_metrics if mode == 'train' else test_metrics
                    for key, value in data.items():
                        if key not in ['epoch', 'mode']:
                            metrics_dict[epoch][key] = value
                
                except json.JSONDecodeError as e:
                    skipped_invalid_json += 1
                    continue
        
        self.metrics['train'] = train_metrics
        self.metrics['test'] = test_metrics
        self.epochs = sorted(list(self.epochs))
        
        print(f"✓ Loaded data for {len(self.epochs)} epochs")
        print(f"  Train metrics: {len(self.metrics['train'])} epochs")
        print(f"  Test metrics: {len(self.metrics['test'])} epochs")
        print(f"  Ignored non-JSON lines: {skipped_non_json}")
        if skipped_invalid_json > 0:
            print(f"  Invalid JSON lines skipped: {skipped_invalid_json}")
    
    def _get_metric_series(self, metric_name: str, mode: str = 'test') -> Tuple[List[int], List[float]]:
        """
        Extract a metric series across epochs.
        
        Args:
            metric_name: Name of the metric (e.g., 'loss', 'recon_acc')
            mode: 'train' or 'test'
            
        Returns:
            (epochs, values) tuples
        """
        metrics_dict = self.metrics[mode]
        epochs = []
        values = []
        
        for epoch in self.epochs:
            if epoch in metrics_dict and metric_name in metrics_dict[epoch]:
                epochs.append(epoch)
                values.append(metrics_dict[epoch][metric_name])
        
        return epochs, values
    
    def _setup_axis(self, ax, xlabel: str = 'Epoch', ylabel: str = '', title: str = ''):
        """Setup common axis formatting."""
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    def _get_first_available_metric(self, mode: str, candidates: List[str]) -> Tuple[Optional[str], List[int], List[float]]:
        """Return the first metric series that exists from a list of candidate names."""
        for name in candidates:
            epochs, values = self._get_metric_series(name, mode=mode)
            if epochs:
                return name, epochs, values
        return None, [], []
    
    def plot_loss(self) -> None:
        """Plot training and test loss over epochs."""
        print("  Generating: Loss plot...")
        
        train_epochs, train_loss = self._get_metric_series('loss', mode='train')
        test_epochs, test_loss = self._get_metric_series('loss', mode='test')
        
        if not train_epochs and not test_epochs:
            print("    ⚠ No loss data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_epochs:
            ax.plot(train_epochs, train_loss, 'o-', label='Train Loss', 
                   linewidth=2, markersize=4, color='#1f77b4', alpha=0.8)
        if test_epochs:
            ax.plot(test_epochs, test_loss, 's-', label='Test Loss', 
                   linewidth=2, markersize=4, color='#ff7f0e', alpha=0.8)
        
        self._setup_axis(ax, ylabel='Loss', title='Training and Test Loss Over Epochs')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: loss.png")
    
    def plot_recon_accuracy(self) -> None:
        """Plot reconstruction accuracy over epochs."""
        print("  Generating: Reconstruction Accuracy plot...")
        
        train_epochs, train_acc = self._get_metric_series('recon_acc', mode='train')
        test_epochs, test_acc = self._get_metric_series('recon_acc', mode='test')
        
        if not train_epochs and not test_epochs:
            print("    ⚠ No recon_acc data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_epochs:
            ax.plot(train_epochs, train_acc, 'o-', label='Train Accuracy', 
                   linewidth=2, markersize=4, color='#2ca02c', alpha=0.8)
        if test_epochs:
            ax.plot(test_epochs, test_acc, 's-', label='Test Accuracy', 
                   linewidth=2, markersize=4, color='#d62728', alpha=0.8)
        
        self._setup_axis(ax, ylabel='Accuracy', title='Entity Reconstruction Accuracy Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'recon_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: recon_accuracy.png")
    
    def plot_mean_reward(self) -> None:
        """Plot mean reward over epochs."""
        print("  Generating: Mean Reward plot...")
        
        train_epochs, train_reward = self._get_metric_series('mean_reward', mode='train')
        test_epochs, test_reward = self._get_metric_series('mean_reward', mode='test')
        
        if not train_epochs and not test_epochs:
            print("    ⚠ No mean_reward data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_epochs:
            ax.plot(train_epochs, train_reward, 'o-', label='Train Reward', 
                   linewidth=2, markersize=4, color='#9467bd', alpha=0.8)
        if test_epochs:
            ax.plot(test_epochs, test_reward, 's-', label='Test Reward', 
                   linewidth=2, markersize=4, color='#8c564b', alpha=0.8)
        
        self._setup_axis(ax, ylabel='Mean Reward', title='Mean Reward Over Epochs')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'mean_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: mean_reward.png")
    
    def plot_recon_loss(self) -> None:
        """Plot reconstruction loss over epochs."""
        print("  Generating: Reconstruction Loss plot...")
        
        train_epochs, train_recon_loss = self._get_metric_series('recon_loss', mode='train')
        test_epochs, test_recon_loss = self._get_metric_series('recon_loss', mode='test')
        
        if not train_epochs and not test_epochs:
            print("    ⚠ No recon_loss data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_epochs:
            ax.plot(train_epochs, train_recon_loss, 'o-', label='Train Recon Loss', 
                   linewidth=2, markersize=4, color='#e377c2', alpha=0.8)
        if test_epochs:
            ax.plot(test_epochs, test_recon_loss, 's-', label='Test Recon Loss', 
                   linewidth=2, markersize=4, color='#7f7f7f', alpha=0.8)
        
        self._setup_axis(ax, ylabel='Reconstruction Loss', title='Reconstruction Loss Over Epochs')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'recon_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: recon_loss.png")
    
    def plot_action_distribution(self) -> None:
        """Plot action distribution rates over epochs."""
        print("  Generating: Action Distribution plot...")
        
        actions = ['hunt_rate', 'gather_rate', 'flee_rate', 'rest_rate', 
                  'mitigate_rate', 'endure_rate', 'eat_rate', 'craft_rate']
        
        # Get test data (more stable)
        action_data = {}
        for action in actions:
            epochs, values = self._get_metric_series(action, mode='test')
            if epochs:
                action_data[action] = (epochs, values)
        
        if not action_data:
            print("    ⚠ No action data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for (action, (epochs, values)), color in zip(action_data.items(), colors):
            label = action.replace('_rate', '').replace('_', ' ').title()
            ax.plot(epochs, values, 'o-', label=label, linewidth=2, markersize=4, 
                   color=color, alpha=0.7)
        
        self._setup_axis(ax, ylabel='Rate', title='Action Distribution Rates Over Epochs')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: action_distribution.png")
    
    def plot_entropy(self) -> None:
        """Plot message entropy over epochs."""
        print("  Generating: Message Entropy plot...")
        
        train_epochs, train_entropy = self._get_metric_series('entropy', mode='train')
        test_epochs, test_entropy = self._get_metric_series('entropy', mode='test')
        
        if not train_epochs and not test_epochs:
            print("    ⚠ No entropy data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_epochs:
            ax.plot(train_epochs, train_entropy, 'o-', label='Train Entropy', 
                   linewidth=2, markersize=4, color='#17becf', alpha=0.8)
        if test_epochs:
            ax.plot(test_epochs, test_entropy, 's-', label='Test Entropy', 
                   linewidth=2, markersize=4, color='#bcbd22', alpha=0.8)
        
        self._setup_axis(ax, ylabel='Shannon Entropy', title='Message Entropy Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: entropy.png")
    
    def plot_topsim(self) -> None:
        """Plot topographic similarity over epochs."""
        print("  Generating: TopSim (Topographic Similarity) plot...")
        
        epochs, values = self._get_metric_series('topsim', mode='test')
        
        if not epochs:
            print("    ⚠ No topsim data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, values, 'o-', label='TopSim', linewidth=2.5, markersize=5, 
               color='#ff7f0e', alpha=0.8)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Mean: {mean_val:.4f}', alpha=0.7)
        
        self._setup_axis(ax, ylabel='TopSim Score', title='Topographic Similarity Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'topsim.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: topsim.png")
    
    def plot_posdis(self) -> None:
        """Plot positional disentanglement over epochs."""
        print("  Generating: PosDis (Positional Disentanglement) plot...")
        
        epochs, values = self._get_metric_series('posdis', mode='test')
        
        if not epochs:
            print("    ⚠ No posdis data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, values, 's-', label='PosDis', linewidth=2.5, markersize=5, 
               color='#2ca02c', alpha=0.8)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Mean: {mean_val:.4f}', alpha=0.7)
        
        self._setup_axis(ax, ylabel='PosDis Score', title='Positional Disentanglement Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'posdis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: posdis.png")
    
    def plot_bosdis(self) -> None:
        """Plot bag-of-symbols disentanglement over epochs."""
        print("  Generating: BosDis (Bag-of-Symbols Disentanglement) plot...")
        
        epochs, values = self._get_metric_series('bosdis', mode='test')
        
        if not epochs:
            print("    ⚠ No bosdis data found, skipping...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, values, '^-', label='BosDis', linewidth=2.5, markersize=5, 
               color='#d62728', alpha=0.8)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='darkorange', linestyle='--', linewidth=1.5, 
                  label=f'Mean: {mean_val:.4f}', alpha=0.7)
        
        self._setup_axis(ax, ylabel='BosDis Score', title='Bag-of-Symbols Disentanglement Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'bosdis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: bosdis.png")
    
    def plot_language_metrics_combined(self) -> None:
        """Plot all language analysis metrics (entropy, topsim, posdis, bosdis) together."""
        print("  Generating: Combined Language Metrics plot...")
        
        # Get test data for all metrics
        entropy_epochs, entropy_vals = self._get_metric_series('entropy', mode='test')
        topsim_epochs, topsim_vals = self._get_metric_series('topsim', mode='test')
        posdis_epochs, posdis_vals = self._get_metric_series('posdis', mode='test')
        bosdis_epochs, bosdis_vals = self._get_metric_series('bosdis', mode='test')
        
        if not any([entropy_epochs, topsim_epochs, posdis_epochs, bosdis_epochs]):
            print("    ⚠ No language metrics data found, skipping...")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Entropy
        if entropy_epochs:
            axes[0, 0].plot(entropy_epochs, entropy_vals, 'o-', linewidth=2, markersize=4, color='#17becf')
            axes[0, 0].set_title('Message Entropy', fontweight='bold')
            axes[0, 0].set_ylabel('Shannon Entropy')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(bottom=0)
        
        # Plot 2: TopSim
        if topsim_epochs:
            axes[0, 1].plot(topsim_epochs, topsim_vals, 's-', linewidth=2, markersize=4, color='#ff7f0e')
            mean_topsim = np.mean(topsim_vals)
            axes[0, 1].axhline(y=mean_topsim, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Topographic Similarity', fontweight='bold')
            axes[0, 1].set_ylabel('TopSim Score')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: PosDis
        if posdis_epochs:
            axes[1, 0].plot(posdis_epochs, posdis_vals, 'd-', linewidth=2, markersize=4, color='#2ca02c')
            mean_posdis = np.mean(posdis_vals)
            axes[1, 0].axhline(y=mean_posdis, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Positional Disentanglement', fontweight='bold')
            axes[1, 0].set_ylabel('PosDis Score')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(bottom=0)
        
        # Plot 4: BosDis
        if bosdis_epochs:
            axes[1, 1].plot(bosdis_epochs, bosdis_vals, '^-', linewidth=2, markersize=4, color='#d62728')
            mean_bosdis = np.mean(bosdis_vals)
            axes[1, 1].axhline(y=mean_bosdis, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Bag-of-Symbols Disentanglement', fontweight='bold')
            axes[1, 1].set_ylabel('BosDis Score')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(bottom=0)
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.suptitle('Language Analysis Metrics Over Epochs', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'language_metrics_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: language_metrics_combined.png")

    def plot_precision(self) -> None:
        """Plot precision over epochs if available in logs."""
        print("  Generating: Precision plot...")
        metric_name, epochs, values = self._get_first_available_metric(
            mode='test',
            candidates=['precision', 'recon_precision', 'macro_precision', 'weighted_precision']
        )

        if not epochs:
            print("    ⚠ No precision metric found in log, skipping...")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, values, 'o-', label=metric_name, linewidth=2.5, markersize=5, color='#1f77b4', alpha=0.85)
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}', alpha=0.7)
        self._setup_axis(ax, ylabel='Precision', title='Precision Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'precision.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: precision.png")

    def plot_recall(self) -> None:
        """Plot recall over epochs if available in logs."""
        print("  Generating: Recall plot...")
        metric_name, epochs, values = self._get_first_available_metric(
            mode='test',
            candidates=['recall', 'recon_recall', 'macro_recall', 'weighted_recall']
        )

        if not epochs:
            print("    ⚠ No recall metric found in log, skipping...")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, values, 'o-', label=metric_name, linewidth=2.5, markersize=5, color='#2ca02c', alpha=0.85)
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}', alpha=0.7)
        self._setup_axis(ax, ylabel='Recall', title='Recall Over Epochs')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: recall.png")

    def _build_confusion_matrix_from_predictions(self) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Build confusion matrix from predictions JSONL if available.

        Expected per line JSON: {"y_true": <int/str>, "y_pred": <int/str>}
        """
        pred_files = sorted(self.run_dir.glob("*predictions*.jsonl"))
        if not pred_files:
            return None, None

        y_true: List[str] = []
        y_pred: List[str] = []
        with open(pred_files[0], 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'y_true' in row and 'y_pred' in row:
                    y_true.append(str(row['y_true']))
                    y_pred.append(str(row['y_pred']))

        if not y_true:
            return None, None

        labels = sorted(set(y_true) | set(y_pred))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            matrix[label_to_idx[t], label_to_idx[p]] += 1

        return matrix, labels

    def _load_confusion_matrix_file(self) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Load confusion matrix from JSON file if available.

        Supports:
        1) {"matrix": [[...]], "labels": [...]} format
        2) [[...], [...]] matrix-only format
        """
        candidates = [
            self.run_dir / 'confusion_matrix.json',
            self.run_dir / 'recon_confusion_matrix.json',
        ]
        for path in candidates:
            if not path.exists():
                continue
            with open(path, 'r') as f:
                payload = json.load(f)

            if isinstance(payload, dict) and 'matrix' in payload:
                matrix = np.array(payload['matrix'])
                labels = [str(x) for x in payload.get('labels', list(range(matrix.shape[0])))]
                return matrix, labels

            if isinstance(payload, list):
                matrix = np.array(payload)
                labels = [str(i) for i in range(matrix.shape[0])]
                return matrix, labels

        return None, None

    def plot_confusion_matrix(self) -> None:
        """Plot confusion matrix if matrix data or prediction pairs are available."""
        print("  Generating: Confusion Matrix plot...")

        matrix, labels = self._load_confusion_matrix_file()
        if matrix is None:
            matrix, labels = self._build_confusion_matrix_from_predictions()

        if matrix is None or labels is None:
            print("    ⚠ No confusion matrix source found (need confusion_matrix.json or *predictions*.jsonl), skipping...")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

        # Keep ticks readable for large label sets
        max_labels = 30
        if len(labels) <= max_labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Annotate only if matrix is reasonably small
        if matrix.shape[0] <= 15 and matrix.shape[1] <= 15:
            threshold = matrix.max() / 2.0 if matrix.size else 0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = int(matrix[i, j])
                    ax.text(j, i, str(val), ha='center', va='center',
                            color='white' if val > threshold else 'black', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: confusion_matrix.png")

    def _find_first_existing(self, patterns: List[str]) -> Optional[Path]:
        """Return first file matching the given glob patterns in run_dir."""
        for pattern in patterns:
            matches = sorted(self.run_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _extract_run_id(name: str) -> Optional[str]:
        return _extract_run_id(name)

    def _resolve_snapshot_paths_for_selected_log(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Resolve progression/final snapshot paths for the currently selected log."""
        log_stem = self.log_file.stem

        progression_path = _resolve_artifact_for_log(
            self.run_dir,
            log_stem,
            artifact_prefix="message_progression",
            extension=".jsonl",
        )
        final_path = _resolve_artifact_for_log(
            self.run_dir,
            log_stem,
            artifact_prefix="message_snapshot_final",
            extension=".json",
        )

        return progression_path, final_path

    def _load_hungarian_snapshots(self) -> Dict[int, Dict]:
        """Load progression + final snapshots for Hungarian analysis."""
        snapshots_by_epoch: Dict[int, Dict] = {}

        progression_path, final_path = self._resolve_snapshot_paths_for_selected_log()

        if progression_path and progression_path.exists():
            with open(progression_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('{'):
                        continue
                    try:
                        snap = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    epoch = snap.get('epoch')
                    if isinstance(epoch, int):
                        snapshots_by_epoch[epoch] = snap

        if final_path and final_path.exists():
            with open(final_path, 'r') as f:
                try:
                    snap = json.load(f)
                except json.JSONDecodeError:
                    snap = {}
            epoch = snap.get('epoch')
            if isinstance(epoch, int):
                snapshots_by_epoch[epoch] = snap

        return snapshots_by_epoch

    @staticmethod
    def _parse_entity_vector(vec_key: str) -> np.ndarray:
        """Parse entity vector key like '1 2 0 3 0 0' into np.array."""
        parts = [int(x) for x in vec_key.strip().split()]
        return np.array(parts, dtype=float)

    @staticmethod
    def _parse_message_tokens(msg: str) -> List[int]:
        """Parse message token string into integer token list."""
        return [int(x) for x in msg.strip().split() if x.strip()]

    def _snapshot_to_count_matrix(self, snapshot: Dict) -> Tuple[List[str], List[str], np.ndarray, List[str], np.ndarray]:
        """Convert snapshot to entity-message count matrix.

        Returns:
            entity_keys, entity_names, entity_vectors, messages, counts_matrix
        """
        by_entity = snapshot.get('by_entity', {})
        if not isinstance(by_entity, dict) or not by_entity:
            return [], [], np.zeros((0, 6)), [], np.zeros((0, 0))

        # Stable order: sort by vector string
        entity_keys = sorted(by_entity.keys())
        entity_names: List[str] = []
        entity_vectors: List[np.ndarray] = []

        # Collect message vocabulary from current snapshot
        all_messages = set()
        for key in entity_keys:
            info = by_entity.get(key, {})
            top_messages = info.get('top_messages', [])
            for tm in top_messages:
                msg = tm.get('message')
                if isinstance(msg, str):
                    all_messages.add(msg)

        messages = sorted(all_messages)
        msg_to_idx = {m: i for i, m in enumerate(messages)}
        counts = np.zeros((len(entity_keys), len(messages)), dtype=float)

        for i, key in enumerate(entity_keys):
            info = by_entity.get(key, {})
            entity_names.append(str(info.get('entity', key)))
            entity_vectors.append(self._parse_entity_vector(key))

            for tm in info.get('top_messages', []):
                msg = tm.get('message')
                cnt = tm.get('count', 0)
                if isinstance(msg, str) and msg in msg_to_idx:
                    try:
                        counts[i, msg_to_idx[msg]] += float(cnt)
                    except (TypeError, ValueError):
                        continue

        return entity_keys, entity_names, np.array(entity_vectors), messages, counts

    @staticmethod
    def _sanitize_message_for_name(message: str) -> str:
        return message.replace(" ", "_")

    def _run_message_snapshot_analysis(self) -> None:
        """Create message/entity heatmap + reuse report from final snapshot."""
        print("  Generating: Message snapshot heatmap and reuse analysis...")
        _progression_path, final_path = self._resolve_snapshot_paths_for_selected_log()
        if not final_path or not final_path.exists():
            print("    ⚠ No final message snapshot found, skipping message heatmap...")
            return

        try:
            with open(final_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except (OSError, json.JSONDecodeError):
            print("    ⚠ Could not parse final message snapshot, skipping message heatmap...")
            return

        entity_keys, entity_names, _entity_vectors, messages, counts = self._snapshot_to_count_matrix(snapshot)
        if not entity_keys or not messages or counts.size == 0:
            print("    ⚠ Snapshot has no entity/message count content, skipping...")
            return

        # Row-normalized matrix makes multi-entity message reuse easier to spot visually.
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized = counts / row_sums

        fig_w = max(12.0, min(26.0, 10.0 + 0.22 * len(messages)))
        fig_h = max(10.0, min(26.0, 8.0 + 0.22 * len(entity_names)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(normalized, aspect="auto", cmap="viridis")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Entity-row normalized message frequency", fontsize=10)

        msg_labels = [self._sanitize_message_for_name(m) for m in messages]
        ax.set_xticks(np.arange(len(messages)))
        ax.set_xticklabels(msg_labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(entity_names)))
        ax.set_yticklabels(entity_names, fontsize=7)
        ax.set_xlabel("Messages", fontsize=11, fontweight="bold")
        ax.set_ylabel("Entities", fontsize=11, fontweight="bold")
        ax.set_title("Message → Entity Heatmap (Final Snapshot)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.metrics_dir / "message_entity_heatmap_final.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Saved: message_entity_heatmap_final.png")

        msg_entity_counts = (counts > 0).sum(axis=0)
        total_by_msg = counts.sum(axis=0)
        reused_indices = [i for i, n in enumerate(msg_entity_counts) if int(n) >= 2]

        # CSV report to explicitly inspect potential synonym messages.
        csv_path = self.metrics_dir / "message_reuse_summary_final.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "message",
                "entities_using_message",
                "total_count",
                "entity_list",
            ])
            for i in sorted(
                range(len(messages)),
                key=lambda j: (int(msg_entity_counts[j]), float(total_by_msg[j])),
                reverse=True,
            ):
                ents = [entity_names[r] for r in range(len(entity_names)) if counts[r, i] > 0]
                writer.writerow([
                    messages[i],
                    int(msg_entity_counts[i]),
                    float(total_by_msg[i]),
                    "|".join(ents),
                ])
        print("    ✓ Saved: message_reuse_summary_final.csv")

        if reused_indices:
            ranked = sorted(
                reused_indices,
                key=lambda j: (int(msg_entity_counts[j]), float(total_by_msg[j])),
                reverse=True,
            )[:20]
            labels = [self._sanitize_message_for_name(messages[j]) for j in ranked]
            values = [int(msg_entity_counts[j]) for j in ranked]

            fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(ranked) + 2)))
            y = np.arange(len(ranked))
            ax.barh(y, values, color="#1f77b4", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Number of entities using this message", fontsize=11, fontweight="bold")
            ax.set_title("Most Reused Messages (Potential Synonyms)", fontsize=12, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(self.metrics_dir / "message_reuse_top20_final.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("    ✓ Saved: message_reuse_top20_final.png")
        else:
            print("    ⚠ No reused messages found in final snapshot (each message maps to one entity).")

    @staticmethod
    def _hungarian_with_padding(cost_matrix: np.ndarray, dummy_cost: float) -> Tuple[np.ndarray, np.ndarray]:
        """Run Hungarian assignment with column padding for full row coverage."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("scipy is required for Hungarian analysis") from exc

        n_rows, n_cols = cost_matrix.shape
        if n_cols < n_rows:
            pad = np.full((n_rows, n_rows - n_cols), dummy_cost, dtype=float)
            padded = np.concatenate([cost_matrix, pad], axis=1)
        else:
            padded = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded)
        return row_ind, col_ind

    def _compute_distance_cost(self, entity_vectors: np.ndarray, messages: List[str], counts: np.ndarray) -> np.ndarray:
        """Build distance-fit cost matrix using message semantic prototypes.

        Message prototype = weighted average entity vector among entities emitting that message.
        Cost(entity, message) = L1 distance to prototype.
        """
        n_entities, n_messages = counts.shape
        if n_entities == 0 or n_messages == 0:
            return np.zeros((n_entities, n_messages), dtype=float)

        # Weighted semantic prototype for each message
        prototypes = np.zeros((n_messages, entity_vectors.shape[1]), dtype=float)
        msg_weights = counts.sum(axis=0)

        for j in range(n_messages):
            w = msg_weights[j]
            if w <= 0:
                # Fallback to token-based weak prior if message has no support
                toks = self._parse_message_tokens(messages[j])
                if toks:
                    token_mean = float(np.mean(toks))
                    prototypes[j, :] = token_mean / 49.0
                continue
            prototypes[j, :] = (counts[:, j][:, None] * entity_vectors).sum(axis=0) / w

        # L1 distance entity -> message prototype
        cost = np.zeros((n_entities, n_messages), dtype=float)
        for i in range(n_entities):
            cost[i, :] = np.abs(prototypes - entity_vectors[i]).sum(axis=1)

        return cost

    def _run_single_hungarian(self, snapshot: Dict, objective: str) -> Optional[Dict]:
        """Run Hungarian analysis for one snapshot and one objective."""
        entity_keys, entity_names, entity_vectors, messages, counts = self._snapshot_to_count_matrix(snapshot)
        n_entities = len(entity_keys)
        n_messages = len(messages)

        if n_entities == 0 or n_messages == 0:
            return None

        if objective == 'frequency':
            max_count = float(np.max(counts)) if counts.size else 1.0
            cost = max_count - counts
            dummy_cost = max_count + 1.0
        else:
            cost = self._compute_distance_cost(entity_vectors, messages, counts)
            max_dist = float(np.max(cost)) if cost.size else 1.0
            dummy_cost = max_dist + 1.0

        row_ind, col_ind = self._hungarian_with_padding(cost, dummy_cost)

        assignments = []
        real_assigned = 0
        matched_count_sum = 0.0
        matched_distance_sum = 0.0

        for r, c in zip(row_ind, col_ind):
            entity_key = entity_keys[r]
            entity_name = entity_names[r]

            if c < n_messages:
                message = messages[c]
                observed_count = float(counts[r, c])
                matched_cost = float(cost[r, c])
                real_assigned += 1
                matched_count_sum += observed_count
                matched_distance_sum += matched_cost
            else:
                message = '__DUMMY__'
                observed_count = 0.0
                matched_cost = float(dummy_cost)

            assignments.append({
                'entity_key': entity_key,
                'entity_name': entity_name,
                'assigned_message': message,
                'count': observed_count,
                'cost': matched_cost,
            })

        total_count = float(counts.sum()) if counts.size else 1.0
        coverage = real_assigned / max(n_entities, 1)

        # Keep objective-specific score semantics simple and interpretable.
        if objective == 'frequency':
            normalized_score = matched_count_sum / max(total_count, 1e-9)
            mean_cost = matched_distance_sum / max(real_assigned, 1)
        else:
            mean_distance = matched_distance_sum / max(real_assigned, 1)
            normalized_score = 1.0 / (1.0 + mean_distance)
            mean_cost = mean_distance

        return {
            'epoch': int(snapshot.get('epoch', -1)),
            'objective': objective,
            'n_entities': n_entities,
            'n_messages': n_messages,
            'coverage': coverage,
            'normalized_score': normalized_score,
            'mean_cost': mean_cost,
            'assignments': assignments,
            'messages': messages,
            'count_matrix': counts.tolist(),
            'entity_names': entity_names,
        }

    def _save_hungarian_outputs(self, all_results: Dict[str, List[Dict]]) -> None:
        """Save Hungarian trajectory plot and summary files."""
        summary_path = self.metrics_dir / 'hungarian_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print("    ✓ Saved: hungarian_summary.json")

        # Trajectory plot
        fig, ax = plt.subplots(figsize=(11, 6))
        has_curve = False
        for objective, color in [('frequency', '#1f77b4'), ('distance', '#ff7f0e')]:
            rows = sorted(all_results.get(objective, []), key=lambda x: x['epoch'])
            if not rows:
                continue
            epochs = [r['epoch'] for r in rows]
            scores = [r['normalized_score'] for r in rows]
            ax.plot(epochs, scores, 'o-', linewidth=2.2, markersize=5,
                    label=f'Hungarian {objective} score', color=color, alpha=0.9)
            has_curve = True

        if has_curve:
            self._setup_axis(ax, ylabel='Normalized Assignment Score', title='Hungarian Assignment Score Over Epochs')
            ax.set_ylim([0, 1.05])
            ax.legend(loc='best', fontsize=10)
            plt.tight_layout()
            plt.savefig(self.metrics_dir / 'hungarian_score_over_epochs.png', dpi=300, bbox_inches='tight')
            print("    ✓ Saved: hungarian_score_over_epochs.png")
        plt.close()

        # Save final-epoch assignment table for both objectives
        csv_path = self.metrics_dir / 'hungarian_assignments_final.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['objective', 'epoch', 'entity_name', 'entity_key', 'assigned_message', 'count', 'cost'])
            for objective in ('frequency', 'distance'):
                rows = sorted(all_results.get(objective, []), key=lambda x: x['epoch'])
                if not rows:
                    continue
                final_row = rows[-1]
                for a in final_row['assignments']:
                    writer.writerow([
                        objective,
                        final_row['epoch'],
                        a['entity_name'],
                        a['entity_key'],
                        a['assigned_message'],
                        f"{a['count']:.6f}",
                        f"{a['cost']:.6f}",
                    ])
        print("    ✓ Saved: hungarian_assignments_final.csv")

        # Final frequency confusion-style heatmap (entity x assigned-message count)
        freq_rows = sorted(all_results.get('frequency', []), key=lambda x: x['epoch'])
        if freq_rows:
            final = freq_rows[-1]
            n_entities = final['n_entities']
            assignments = final['assignments']
            entity_labels = [a['entity_name'] for a in assignments]
            msg_labels = [a['assigned_message'] for a in assignments]
            values = np.array([a['count'] for a in assignments], dtype=float)

            # diagonal matrix to present assignment support per entity
            heat = np.zeros((n_entities, n_entities), dtype=float)
            np.fill_diagonal(heat, values)

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(heat, interpolation='nearest', cmap='YlGnBu')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Hungarian Assignment Support (Final Epoch, Frequency Objective)', fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Assigned Message Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Entity', fontsize=11, fontweight='bold')

            ax.set_xticks(np.arange(n_entities))
            ax.set_yticks(np.arange(n_entities))
            ax.set_xticklabels([str(i + 1) for i in range(n_entities)], rotation=90, fontsize=7)
            ax.set_yticklabels(entity_labels, fontsize=7)

            plt.tight_layout()
            plt.savefig(self.metrics_dir / 'hungarian_assignment_heatmap_final.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("    ✓ Saved: hungarian_assignment_heatmap_final.png")

            # Also store message index legend for interpretability
            legend_path = self.metrics_dir / 'hungarian_message_index_legend.csv'
            with open(legend_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'entity_name', 'assigned_message'])
                for i, (entity, msg) in enumerate(zip(entity_labels, msg_labels), start=1):
                    writer.writerow([i, entity, msg])
            print("    ✓ Saved: hungarian_message_index_legend.csv")

    def run_hungarian_analysis(self) -> None:
        """Run Hungarian assignment analysis over checkpoints from message outputs."""
        print("  Generating: Hungarian assignment analysis...")

        snapshots_by_epoch = self._load_hungarian_snapshots()
        if not snapshots_by_epoch:
            print("    ⚠ No message progression/final snapshot files found, skipping Hungarian analysis...")
            return

        target_epochs = sorted(set(self.hungarian_epochs + [max(snapshots_by_epoch.keys())]))
        available_epochs = [e for e in target_epochs if e in snapshots_by_epoch]
        if not available_epochs:
            print("    ⚠ No requested Hungarian epochs available in snapshots, skipping...")
            return

        objectives = ['frequency', 'distance'] if self.hungarian_mode == 'both' else [self.hungarian_mode]
        all_results: Dict[str, List[Dict]] = {obj: [] for obj in objectives}

        for epoch in available_epochs:
            snap = snapshots_by_epoch[epoch]
            for obj in objectives:
                result = self._run_single_hungarian(snap, objective=obj)
                if result is not None:
                    all_results[obj].append(result)

        # Ensure deterministic ordering
        for obj in objectives:
            all_results[obj] = sorted(all_results[obj], key=lambda x: x['epoch'])

        self.hungarian_results = all_results
        self._save_hungarian_outputs(all_results)
    
    def generate_summary_report(self) -> None:
        """Generate a text summary of metrics."""
        print("  Generating: Summary report...")
        
        report_path = self.metrics_dir / 'metrics_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("METRICS SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write(f"Total Epochs: {len(self.epochs)}\n\n")
            
            # Loss metrics
            f.write("LOSS METRICS\n")
            f.write("-" * 70 + "\n")
            test_epochs, test_loss = self._get_metric_series('loss', mode='test')
            if test_epochs:
                f.write(f"Test Loss - Final: {test_loss[-1]:.6f}, Min: {min(test_loss):.6f}, Max: {max(test_loss):.6f}\n")
            
            # Recon accuracy
            f.write("\nRECONSTRUCTION ACCURACY\n")
            f.write("-" * 70 + "\n")
            test_epochs, test_acc = self._get_metric_series('recon_acc', mode='test')
            if test_epochs:
                f.write(f"Test Accuracy - Final: {test_acc[-1]:.6f}, Min: {min(test_acc):.6f}, Max: {max(test_acc):.6f}\n")
            
            # Mean reward
            f.write("\nMEAN REWARD\n")
            f.write("-" * 70 + "\n")
            test_epochs, test_reward = self._get_metric_series('mean_reward', mode='test')
            if test_epochs:
                f.write(f"Test Reward - Final: {test_reward[-1]:.6f}, Mean: {np.mean(test_reward):.6f}\n")
            
            # Language metrics
            f.write("\nLANGUAGE ANALYSIS METRICS\n")
            f.write("-" * 70 + "\n")
            
            epochs, entropy = self._get_metric_series('entropy', mode='test')
            if epochs:
                f.write(f"Entropy - Final: {entropy[-1]:.6f}, Mean: {np.mean(entropy):.6f}\n")
            
            epochs, topsim = self._get_metric_series('topsim', mode='test')
            if epochs:
                f.write(f"TopSim - Final: {topsim[-1]:.6f}, Mean: {np.mean(topsim):.6f}\n")
            
            epochs, posdis = self._get_metric_series('posdis', mode='test')
            if epochs:
                f.write(f"PosDis - Final: {posdis[-1]:.6f}, Mean: {np.mean(posdis):.6f}\n")
            
            epochs, bosdis = self._get_metric_series('bosdis', mode='test')
            if epochs:
                f.write(f"BosDis - Final: {bosdis[-1]:.6f}, Mean: {np.mean(bosdis):.6f}\n")

            precision_name, p_epochs, p_values = self._get_first_available_metric(
                mode='test', candidates=['precision', 'recon_precision', 'macro_precision', 'weighted_precision']
            )
            if p_epochs:
                f.write(
                    f"{precision_name} - Final: {p_values[-1]:.6f}, Mean: {np.mean(p_values):.6f}\n"
                )

            recall_name, r_epochs, r_values = self._get_first_available_metric(
                mode='test', candidates=['recall', 'recon_recall', 'macro_recall', 'weighted_recall']
            )
            if r_epochs:
                f.write(
                    f"{recall_name} - Final: {r_values[-1]:.6f}, Mean: {np.mean(r_values):.6f}\n"
                )

            if self.hungarian_results:
                f.write("\nHUNGARIAN ASSIGNMENT METRICS\n")
                f.write("-" * 70 + "\n")
                for objective in ('frequency', 'distance'):
                    rows = self.hungarian_results.get(objective, [])
                    if not rows:
                        continue
                    final = rows[-1]
                    f.write(
                        f"{objective.title()} - Final epoch: {final['epoch']}, "
                        f"Score: {final['normalized_score']:.6f}, "
                        f"Coverage: {final['coverage']:.6f}, "
                        f"Mean cost: {final['mean_cost']:.6f}\n"
                    )
            
            f.write("\n" + "=" * 70 + "\n")
        
        print("    ✓ Saved: metrics_summary.txt")
    
    def generate_all(self) -> None:
        """Generate all metrics visualizations."""
        print("\n[Generating visualizations...]")
        
        self.plot_loss()
        self.plot_recon_accuracy()
        self.plot_mean_reward()
        self.plot_recon_loss()
        self.plot_action_distribution()
        self.plot_entropy()
        self.plot_topsim()
        self.plot_posdis()
        self.plot_bosdis()
        self.plot_language_metrics_combined()
        self.plot_precision()
        self.plot_recall()
        self.plot_confusion_matrix()
        self.run_hungarian_analysis()
        self._run_message_snapshot_analysis()
        self.generate_summary_report()
        
        print("\n✓ All metrics generated successfully!")
        print(f"✓ Results saved to: {self.metrics_dir}")


class MultiRunMetricsAggregator:
    """
    Aggregate metrics across multiple run logs in one directory.

    Expected filenames include patterns such as train_run14.txt or
    train_msg_len_2_gs_temp_2.0_action_ent_0.3_action_temp_2.0_run24.txt.
    """

    def __init__(self, run_dir: Path, log_pattern: str = "train*"):
        self.run_dir = Path(run_dir).resolve()
        self.metrics_dir = self.run_dir / "metrics"
        self.log_pattern = log_pattern
        self.run_payloads: Dict[str, Dict] = {}
        self.run_snapshots: Dict[str, Dict] = {}

    def _discover_logs(self) -> List[Path]:
        stem_pattern = self.log_pattern
        if stem_pattern.endswith(".log") or stem_pattern.endswith(".txt"):
            candidates = sorted(self.run_dir.glob(stem_pattern))
            return [p for p in candidates if p.is_file()]

        wildcard_candidates = sorted(self.run_dir.glob(stem_pattern))
        filtered = [
            p for p in wildcard_candidates
            if p.is_file() and p.suffix in {".log", ".txt"}
        ]
        if filtered:
            return filtered

        # Backward-compatible fallback for stem-style patterns.
        log_candidates = sorted(self.run_dir.glob(f"{stem_pattern}.log"))
        txt_candidates = sorted(self.run_dir.glob(f"{stem_pattern}.txt"))
        by_name: Dict[str, Path] = {}
        for p in log_candidates + txt_candidates:
            by_name[p.name] = p
        return [by_name[k] for k in sorted(by_name.keys())]

    @staticmethod
    def _parse_log_file(log_file: Path) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], List[int]]:
        train_metrics: Dict[int, Dict[str, float]] = {}
        test_metrics: Dict[int, Dict[str, float]] = {}
        epochs = set()

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped or not line_stripped.startswith("{"):
                    continue

                try:
                    data = json.loads(line_stripped)
                except json.JSONDecodeError:
                    continue

                epoch = data.get("epoch")
                mode = data.get("mode")
                if epoch is None or mode not in ("train", "test"):
                    continue

                epoch = int(epoch)
                epochs.add(epoch)
                target = train_metrics if mode == "train" else test_metrics
                target.setdefault(epoch, {})
                for k, v in data.items():
                    if k in ("epoch", "mode"):
                        continue
                    target[epoch][k] = v

        return {"train": train_metrics, "test": test_metrics}, sorted(epochs)

    @staticmethod
    def _sample_std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        return float(np.std(np.array(values, dtype=np.float64), ddof=1))

    @staticmethod
    def _extract_run_id(name: str) -> Optional[str]:
        return _extract_run_id(name)

    def _load_snapshot_for_log(self, log_file: Path) -> Optional[Dict]:
        snapshot_path = _resolve_artifact_for_log(
            self.run_dir,
            log_file.stem,
            artifact_prefix="message_snapshot_final",
            extension=".json",
        )
        if snapshot_path is None:
            return None
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _snapshot_to_count_matrix(snapshot: Dict) -> Tuple[List[str], List[str], List[str], np.ndarray]:
        by_entity = snapshot.get("by_entity", {})
        if not isinstance(by_entity, dict) or not by_entity:
            return [], [], [], np.zeros((0, 0), dtype=float)

        entity_keys = sorted(by_entity.keys())
        entity_names: List[str] = []
        all_messages = set()

        for k in entity_keys:
            info = by_entity.get(k, {})
            entity_names.append(str(info.get("entity", k)))
            for tm in info.get("top_messages", []):
                msg = tm.get("message")
                if isinstance(msg, str):
                    all_messages.add(msg)

        messages = sorted(all_messages)
        msg_to_idx = {m: i for i, m in enumerate(messages)}
        counts = np.zeros((len(entity_keys), len(messages)), dtype=float)

        for i, k in enumerate(entity_keys):
            info = by_entity.get(k, {})
            for tm in info.get("top_messages", []):
                msg = tm.get("message")
                cnt = tm.get("count", 0)
                if isinstance(msg, str) and msg in msg_to_idx:
                    try:
                        counts[i, msg_to_idx[msg]] += float(cnt)
                    except (TypeError, ValueError):
                        continue

        return entity_keys, entity_names, messages, counts

    @staticmethod
    def _sanitize_message_for_name(message: str) -> str:
        return message.replace(" ", "_")

    def load_all(self) -> None:
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        logs = self._discover_logs()
        if not logs:
            raise FileNotFoundError(
                f"No run files found in {self.run_dir} using pattern '{self.log_pattern}'"
            )

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Aggregate mode: found {len(logs)} run files")
        for log in logs:
            payload, epochs = self._parse_log_file(log)
            snapshot = self._load_snapshot_for_log(log)
            self.run_payloads[log.name] = {
                "file": log,
                "metrics": payload,
                "epochs": epochs,
            }
            if snapshot is not None:
                self.run_snapshots[log.name] = snapshot
            print(f"  - {log.name}: {len(payload['test'])} test epochs")

    def _plot_run_snapshot_heatmap(self, run_name: str, snapshot: Dict) -> None:
        _entity_keys, entity_names, messages, counts = self._snapshot_to_count_matrix(snapshot)
        if not entity_names or not messages or counts.size == 0:
            print(f"    ⚠ {run_name}: snapshot missing message/entity counts, skipping heatmap")
            return

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized = counts / row_sums

        fig_w = max(11.0, min(24.0, 9.0 + 0.20 * len(messages)))
        fig_h = max(9.0, min(24.0, 7.0 + 0.20 * len(entity_names)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(normalized, aspect="auto", cmap="viridis")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Entity-row normalized message frequency", fontsize=10)

        msg_labels = [self._sanitize_message_for_name(m) for m in messages]
        ax.set_xticks(np.arange(len(messages)))
        ax.set_xticklabels(msg_labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(entity_names)))
        ax.set_yticklabels(entity_names, fontsize=7)
        ax.set_xlabel("Messages", fontsize=11, fontweight="bold")
        ax.set_ylabel("Entities", fontsize=11, fontweight="bold")
        ax.set_title(f"Message → Entity Heatmap ({run_name})", fontsize=12, fontweight="bold")
        plt.tight_layout()

        run_label = _stable_run_label(run_name)
        out_name = f"message_entity_heatmap_{run_label}.png"
        plt.savefig(self.metrics_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved: {out_name}")

    def _save_message_reuse_group_summary(self) -> None:
        rows: List[Dict[str, object]] = []

        for run_name, snapshot in self.run_snapshots.items():
            _entity_keys, entity_names, messages, counts = self._snapshot_to_count_matrix(snapshot)
            if not entity_names or not messages or counts.size == 0:
                continue

            msg_entity_counts = (counts > 0).sum(axis=0)
            total_by_msg = counts.sum(axis=0)
            reused = [i for i, n in enumerate(msg_entity_counts) if int(n) >= 2]

            run_label = _stable_run_label(run_name)
            rows.append({
                "run": run_name,
                "run_label": run_label,
                "n_entities": len(entity_names),
                "n_messages": len(messages),
                "reused_messages": len(reused),
                "reuse_ratio": (len(reused) / max(len(messages), 1)),
                "max_entities_on_one_message": int(np.max(msg_entity_counts)) if len(messages) > 0 else 0,
                "top_reused_message": messages[int(np.argmax(msg_entity_counts))] if len(messages) > 0 else "",
                "top_reused_message_entity_count": int(np.max(msg_entity_counts)) if len(messages) > 0 else 0,
                "top_reused_message_total_count": float(np.max(total_by_msg)) if len(messages) > 0 else 0.0,
            })

            # Per-run detailed csv for manual synonym inspection.
            detail_csv = self.metrics_dir / f"message_reuse_summary_{run_label}.csv"
            with open(detail_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "message",
                    "entities_using_message",
                    "total_count",
                    "entity_list",
                ])
                for i in sorted(
                    range(len(messages)),
                    key=lambda j: (int(msg_entity_counts[j]), float(total_by_msg[j])),
                    reverse=True,
                ):
                    ents = [entity_names[r] for r in range(len(entity_names)) if counts[r, i] > 0]
                    writer.writerow([
                        messages[i],
                        int(msg_entity_counts[i]),
                        float(total_by_msg[i]),
                        "|".join(ents),
                    ])
            print(f"    ✓ Saved: message_reuse_summary_{run_label}.csv")

        if not rows:
            print("    ⚠ No snapshot reuse summaries generated (no valid snapshots found).")
            return

        summary_csv = self.metrics_dir / "message_reuse_summary_across_runs.csv"
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "run",
                "run_label",
                "n_entities",
                "n_messages",
                "reused_messages",
                "reuse_ratio",
                "max_entities_on_one_message",
                "top_reused_message",
                "top_reused_message_entity_count",
                "top_reused_message_total_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("    ✓ Saved: message_reuse_summary_across_runs.csv")

        # Bar chart: count of reused messages per run for quick comparison.
        run_labels = [r["run_label"] for r in rows]
        reused_vals = [float(r["reused_messages"]) for r in rows]
        ratio_vals = [float(r["reuse_ratio"]) for r in rows]

        x = np.arange(len(run_labels))
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(x, reused_vals, color="#1f77b4", alpha=0.85, label="Reused messages (count)")
        ax1.set_ylabel("Reused messages", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(run_labels, rotation=25, ha="right")
        ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

        ax2 = ax1.twinx()
        ax2.plot(x, ratio_vals, color="#ff7f0e", marker="o", linewidth=2.0, label="Reuse ratio")
        ax2.set_ylabel("Reuse ratio", fontsize=11, fontweight="bold")
        ax2.set_ylim(0.0, min(1.0, max(ratio_vals) * 1.2 + 0.05))

        ax1.set_title("Message Reuse Across Runs (Potential Synonyms)", fontsize=12, fontweight="bold")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        plt.tight_layout()
        plt.savefig(self.metrics_dir / "message_reuse_across_runs.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Saved: message_reuse_across_runs.png")

    def _final_epoch_metrics(self) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        metric_map: Dict[str, Dict[str, float]] = {}
        run_names = sorted(self.run_payloads.keys())

        for run_name in run_names:
            payload = self.run_payloads[run_name]
            test_metrics: Dict[int, Dict[str, float]] = payload["metrics"]["test"]
            if not test_metrics:
                continue

            final_epoch = max(test_metrics.keys())
            final_row = test_metrics.get(final_epoch, {})
            for key, value in final_row.items():
                metric_map.setdefault(key, {})[run_name] = float(value)

        return run_names, metric_map

    def _save_final_summary(self) -> None:
        run_names, metric_map = self._final_epoch_metrics()
        report_path = self.metrics_dir / "multi_run_summary.txt"
        json_path = self.metrics_dir / "multi_run_summary.json"

        selected_metrics = [
            "loss",
            "mean_reward",
            "expected_reward",
            "recon_loss",
            "recon_acc",
            "topsim",
            "length",
            "valid_action_rate",
            "invalid_entity_idx_rate",
        ]

        summary_json = {
            "run_dir": str(self.run_dir),
            "runs": run_names,
            "run_labels": {rn: _stable_run_label(rn) for rn in run_names},
            "n_runs": len(run_names),
            "metrics": {},
        }

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("MULTI-RUN METRICS SUMMARY (FINAL TEST EPOCH)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Runs: {len(run_names)}\n")
            f.write("Files:\n")
            for rn in run_names:
                f.write(f"  - {rn}\n")
            f.write("\n")

            for metric in selected_metrics:
                per_run = metric_map.get(metric, {})
                values = [per_run[rn] for rn in run_names if rn in per_run]
                if not values:
                    continue

                mean_val = float(np.mean(values))
                std_val = self._sample_std(values)
                f.write(f"{metric}: mean={mean_val:.6f}, std={std_val:.6f}, n={len(values)}\n")
                summary_json["metrics"][metric] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(values),
                    "per_run": {rn: per_run.get(rn) for rn in run_names if rn in per_run},
                }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary_json, jf, indent=2)

        # Wide per-run table for thesis-ready reporting.
        final_table_csv = self.metrics_dir / "multi_run_final_metrics.csv"
        available_metrics = [m for m in selected_metrics if m in metric_map]
        with open(final_table_csv, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["run", "run_label"] + available_metrics)
            for rn in run_names:
                row = [rn, _stable_run_label(rn)]
                for metric in available_metrics:
                    value = metric_map.get(metric, {}).get(rn)
                    row.append("" if value is None else f"{float(value):.6f}")
                writer.writerow(row)

        topsim_per_run = metric_map.get("topsim", {})
        if topsim_per_run:
            topsim_rank_csv = self.metrics_dir / "topsim_ranking.csv"
            ranked = sorted(
                topsim_per_run.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            with open(topsim_rank_csv, "w", encoding="utf-8", newline="") as f_rank:
                writer = csv.writer(f_rank)
                writer.writerow(["rank", "run", "run_label", "topsim"])
                for idx, (rn, value) in enumerate(ranked, start=1):
                    writer.writerow([idx, rn, _stable_run_label(rn), f"{float(value):.6f}"])

        print("    ✓ Saved: multi_run_summary.txt")
        print("    ✓ Saved: multi_run_summary.json")
        print("    ✓ Saved: multi_run_final_metrics.csv")
        if topsim_per_run:
            print("    ✓ Saved: topsim_ranking.csv")

    def _plot_aggregate_metric(self, metric_name: str, out_name: str) -> None:
        valid_epochs, means, stds = self._aggregate_metric_series(metric_name)
        if not valid_epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        means_np = np.array(means)
        stds_np = np.array(stds)
        epochs_np = np.array(valid_epochs)

        ax.plot(epochs_np, means_np, color="#1f77b4", linewidth=2.5, label=f"{metric_name} mean")
        ax.fill_between(
            epochs_np,
            means_np - stds_np,
            means_np + stds_np,
            color="#1f77b4",
            alpha=0.2,
            label="mean ± std",
        )
        ax.set_title(f"{metric_name} across runs (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.metrics_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved: {out_name}")

    def _aggregate_metric_series(self, metric_name: str) -> Tuple[List[int], List[float], List[float]]:
        # Build union of epochs across runs.
        all_epochs = sorted({
            ep
            for payload in self.run_payloads.values()
            for ep in payload["metrics"]["test"].keys()
        })
        if not all_epochs:
            return

        means = []
        stds = []
        valid_epochs = []

        for ep in all_epochs:
            vals = []
            for payload in self.run_payloads.values():
                row = payload["metrics"]["test"].get(ep, {})
                if metric_name in row:
                    vals.append(float(row[metric_name]))
            if not vals:
                continue
            valid_epochs.append(ep)
            means.append(float(np.mean(vals)))
            stds.append(self._sample_std(vals))

        return valid_epochs, means, stds

    def generate(self) -> None:
        print("\n[Generating multi-run aggregate metrics...]")
        self.load_all()
        self._plot_aggregate_metric("loss", "aggregate_loss_test.png")
        self._plot_aggregate_metric("mean_reward", "aggregate_mean_reward_test.png")
        self._plot_aggregate_metric("expected_reward", "aggregate_expected_reward_test.png")
        self._plot_aggregate_metric("recon_acc", "aggregate_recon_acc_test.png")
        self._plot_aggregate_metric("topsim", "aggregate_topsim_test.png")
        self._plot_aggregate_metric("length", "aggregate_length_test.png")
        for run_name in sorted(self.run_snapshots.keys()):
            self._plot_run_snapshot_heatmap(run_name, self.run_snapshots[run_name])
        self._save_message_reuse_group_summary()
        self._save_final_summary()
        print("\n✓ Multi-run aggregation completed")
        print(f"✓ Results saved to: {self.metrics_dir}")


class GroupComparisonAggregator:
    """
    Compare multiple named run groups into one thesis-ready bundle.

    Group spec format:
        --group <name>=<subdir>[:<pattern>]

    Examples:
        --group with_recon=with_recon
        --group no_recon=no_recon:train*
        --group tuning_len2_ae03=no_recon_tuning:train_msg_len_2_gs_temp_2.0_action_ent_0.3_action_temp_2.0_run*
    """

    def __init__(self, root_dir: Path, group_specs: List[str]):
        self.root_dir = Path(root_dir).resolve()
        self.metrics_dir = self.root_dir / "metrics"
        self.group_specs = list(group_specs or [])
        self.group_order: List[str] = []
        self.groups: List[Dict[str, object]] = []
        self._parse_group_specs()

    @staticmethod
    def _parse_single_group_spec(raw_spec: str) -> Tuple[str, str, str]:
        if "=" not in raw_spec:
            raise ValueError(
                f"Invalid --group value '{raw_spec}'. Expected format: name=subdir[:pattern]"
            )

        name, rhs = raw_spec.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        if not name or not rhs:
            raise ValueError(
                f"Invalid --group value '{raw_spec}'. Expected format: name=subdir[:pattern]"
            )

        if ":" in rhs:
            rel_dir, pattern = rhs.split(":", 1)
            rel_dir = rel_dir.strip()
            pattern = pattern.strip() or "train*"
        else:
            rel_dir = rhs
            pattern = "train*"

        rel_dir = rel_dir or "."
        return name, rel_dir, pattern

    def _resolve_group_dir(self, rel_dir: str) -> Path:
        path = Path(rel_dir)
        if path.is_absolute():
            return path.resolve()
        return (self.root_dir / path).resolve()

    def _parse_group_specs(self) -> None:
        if not self.group_specs:
            raise ValueError(
                "--compare-groups requires at least two --group values (name=subdir[:pattern])."
            )

        names = set()
        for spec in self.group_specs:
            name, rel_dir, pattern = self._parse_single_group_spec(spec)
            if name in names:
                raise ValueError(f"Duplicate group name in --group: '{name}'")
            names.add(name)

            run_dir = self._resolve_group_dir(rel_dir)
            aggregator = MultiRunMetricsAggregator(run_dir, log_pattern=pattern)
            self.groups.append({
                "name": name,
                "rel_dir": rel_dir,
                "run_dir": run_dir,
                "pattern": pattern,
                "aggregator": aggregator,
            })
            self.group_order.append(name)

        if len(self.groups) < 2:
            raise ValueError("--compare-groups requires at least two --group entries.")

    @staticmethod
    def _selected_metrics() -> List[str]:
        return [
            "topsim",
            "mean_reward",
            "expected_reward",
            "loss",
            "recon_acc",
            "length",
            "valid_action_rate",
            "invalid_entity_idx_rate",
        ]

    def load_all(self) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Group compare mode: {len(self.groups)} groups")

        for group in self.groups:
            name = str(group["name"])
            run_dir = group["run_dir"]
            pattern = str(group["pattern"])
            aggregator = group["aggregator"]
            print(f"  - Loading group '{name}' from {run_dir} (pattern='{pattern}')")
            aggregator.load_all()

    def _build_summary(self, metrics: List[str]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        summary: Dict[str, object] = {
            "root_dir": str(self.root_dir),
            "n_groups": len(self.groups),
            "groups": {},
        }
        per_run_rows: List[Dict[str, object]] = []

        groups_summary = summary["groups"]
        for group in self.groups:
            name = str(group["name"])
            pattern = str(group["pattern"])
            run_dir = group["run_dir"]
            aggregator = group["aggregator"]

            run_names, metric_map = aggregator._final_epoch_metrics()
            run_labels = {rn: _stable_run_label(rn) for rn in run_names}

            metric_summary: Dict[str, Dict[str, object]] = {}
            for metric in metrics:
                per_run = metric_map.get(metric, {})
                values = [float(per_run[rn]) for rn in run_names if rn in per_run]
                if not values:
                    continue
                metric_summary[metric] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": MultiRunMetricsAggregator._sample_std(values),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "per_run": {rn: float(per_run[rn]) for rn in run_names if rn in per_run},
                }

            groups_summary[name] = {
                "run_dir": str(run_dir),
                "log_pattern": pattern,
                "n_runs": len(run_names),
                "runs": run_names,
                "run_labels": run_labels,
                "metrics": metric_summary,
            }

            for rn in run_names:
                row = {
                    "group": name,
                    "run": rn,
                    "run_label": run_labels.get(rn, rn),
                }
                for metric in metrics:
                    value = metric_map.get(metric, {}).get(rn)
                    row[metric] = None if value is None else float(value)
                per_run_rows.append(row)

        return summary, per_run_rows

    def _write_summary_files(
        self,
        summary: Dict[str, object],
        per_run_rows: List[Dict[str, object]],
        metrics: List[str],
    ) -> None:
        txt_path = self.metrics_dir / "group_compare_summary.txt"
        json_path = self.metrics_dir / "group_compare_summary.json"
        per_run_csv = self.metrics_dir / "group_compare_final_metrics_per_run.csv"
        by_group_csv = self.metrics_dir / "group_compare_final_metrics_by_group.csv"

        groups = summary.get("groups", {})

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("GROUP COMPARISON METRICS SUMMARY (FINAL TEST EPOCH)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Root Directory: {self.root_dir}\n")
            f.write(f"Groups: {len(self.group_order)}\n\n")

            for group_name in self.group_order:
                group_info = groups.get(group_name, {})
                f.write(f"[{group_name}]\n")
                f.write(f"  run_dir: {group_info.get('run_dir')}\n")
                f.write(f"  pattern: {group_info.get('log_pattern')}\n")
                f.write(f"  n_runs: {group_info.get('n_runs')}\n")

                metrics_info = group_info.get("metrics", {})
                for metric in metrics:
                    metric_stats = metrics_info.get(metric)
                    if not metric_stats:
                        continue
                    f.write(
                        f"  {metric}: mean={float(metric_stats['mean']):.6f}, "
                        f"std={float(metric_stats['std']):.6f}, "
                        f"n={int(metric_stats['n'])}\n"
                    )
                f.write("\n")

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)

        with open(per_run_csv, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["group", "run", "run_label"] + metrics)
            for row in per_run_rows:
                values = [
                    ""
                    if row.get(metric) is None
                    else f"{float(row[metric]):.6f}"
                    for metric in metrics
                ]
                writer.writerow([
                    row.get("group", ""),
                    row.get("run", ""),
                    row.get("run_label", ""),
                ] + values)

        with open(by_group_csv, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["group", "metric", "n", "mean", "std", "min", "max"])

            for group_name in self.group_order:
                group_info = groups.get(group_name, {})
                metrics_info = group_info.get("metrics", {})
                for metric in metrics:
                    metric_stats = metrics_info.get(metric)
                    if not metric_stats:
                        continue
                    writer.writerow([
                        group_name,
                        metric,
                        int(metric_stats["n"]),
                        f"{float(metric_stats['mean']):.6f}",
                        f"{float(metric_stats['std']):.6f}",
                        f"{float(metric_stats['min']):.6f}",
                        f"{float(metric_stats['max']):.6f}",
                    ])

        print("    ✓ Saved: group_compare_summary.txt")
        print("    ✓ Saved: group_compare_summary.json")
        print("    ✓ Saved: group_compare_final_metrics_per_run.csv")
        print("    ✓ Saved: group_compare_final_metrics_by_group.csv")

    def _plot_group_metric_bars(self, summary: Dict[str, object], metric_name: str) -> None:
        groups = summary.get("groups", {})
        labels: List[str] = []
        means: List[float] = []
        stds: List[float] = []

        for group_name in self.group_order:
            group_info = groups.get(group_name, {})
            metric_info = group_info.get("metrics", {}).get(metric_name)
            if not metric_info:
                continue
            labels.append(group_name)
            means.append(float(metric_info["mean"]))
            stds.append(float(metric_info["std"]))

        if not labels:
            return

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color="#1f77b4", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
        ax.set_title(
            f"Group comparison: final-epoch {metric_name} (test)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        out_name = f"group_compare_final_{metric_name}.png"
        plt.savefig(self.metrics_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved: {out_name}")

    def _plot_group_trajectory(self, metric_name: str, out_name: str) -> None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.groups))))
        fig, ax = plt.subplots(figsize=(10, 6))

        plotted = False
        for idx, group in enumerate(self.groups):
            name = str(group["name"])
            aggregator = group["aggregator"]
            epochs, means, stds = aggregator._aggregate_metric_series(metric_name)
            if not epochs:
                continue

            plotted = True
            epochs_np = np.array(epochs)
            means_np = np.array(means)
            stds_np = np.array(stds)
            color = colors[idx]

            ax.plot(epochs_np, means_np, color=color, linewidth=2.0, label=name)
            ax.fill_between(
                epochs_np,
                means_np - stds_np,
                means_np + stds_np,
                color=color,
                alpha=0.15,
            )

        if not plotted:
            plt.close()
            return

        ax.set_title(f"Group comparison trajectory: {metric_name} (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.metrics_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved: {out_name}")

    def generate(self) -> None:
        print("\n[Generating grouped comparison metrics...]")
        self.load_all()

        metrics = self._selected_metrics()
        summary, per_run_rows = self._build_summary(metrics)
        self._write_summary_files(summary, per_run_rows, metrics)

        for metric in metrics:
            self._plot_group_metric_bars(summary, metric)

        self._plot_group_trajectory("topsim", "group_compare_topsim_test.png")
        self._plot_group_trajectory("mean_reward", "group_compare_mean_reward_test.png")
        self._plot_group_trajectory("expected_reward", "group_compare_expected_reward_test.png")
        self._plot_group_trajectory("length", "group_compare_length_test.png")

        print("\n✓ Grouped comparison completed")
        print(f"✓ Results saved to: {self.metrics_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive metrics visualizations from training logs"
    )
    parser.add_argument(
        'run_dir',
        type=str,
        help='Path to the run directory containing train_run.log'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional explicit log filename, e.g. train_run13.log'
    )
    parser.add_argument(
        '--hungarian-mode',
        type=str,
        default='both',
        choices=['frequency', 'distance', 'both'],
        help='Hungarian objective mode for message-semantics analysis (default: both)'
    )
    parser.add_argument(
        '--hungarian-epochs',
        type=str,
        default='10,20,30,40,50',
        help='Comma-separated epochs for Hungarian analysis (default: 10,20,30,40,50)'
    )
    parser.add_argument(
        '--all-runs',
        action='store_true',
        help='Aggregate all run files in run_dir (mean/std across runs)'
    )
    parser.add_argument(
        '--compare-groups',
        action='store_true',
        help='Compare multiple run groups via repeated --group entries'
    )
    parser.add_argument(
        '--group',
        action='append',
        default=[],
        help='Group spec for --compare-groups: name=subdir[:pattern] (repeatable)'
    )
    parser.add_argument(
        '--log-pattern',
        type=str,
        default='train*',
        help='Filename glob or stem used for run discovery in --all-runs mode (default: train*)'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("SURVIVAL GAME - METRICS GENERATOR")
        print("=" * 70)

        if args.compare_groups and args.all_runs:
            raise ValueError("Use either --all-runs or --compare-groups, not both.")

        if args.compare_groups:
            comparator = GroupComparisonAggregator(
                args.run_dir,
                group_specs=args.group,
            )
            comparator.generate()
        elif args.all_runs:
            aggregator = MultiRunMetricsAggregator(
                args.run_dir,
                log_pattern=args.log_pattern,
            )
            aggregator.generate()
        else:
            generator = MetricsGenerator(
                args.run_dir,
                log_file=args.log_file,
                hungarian_mode=args.hungarian_mode,
                hungarian_epochs=args.hungarian_epochs,
            )
            generator.load_metrics()
            generator.generate_all()
        
        print("\n" + "=" * 70)
        print("Process completed successfully!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
