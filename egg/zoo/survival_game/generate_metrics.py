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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


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
            return (self.run_dir / explicit_log_file).resolve()

        # Common default first
        default_log = self.run_dir / "train_run.log"
        if default_log.exists():
            return default_log

        # Any run-specific log, e.g. train_run13.log
        candidates = sorted(self.run_dir.glob("train_run*.log"))
        if candidates:
            return candidates[0]

        # Let validation raise a clear error
        return default_log
        
    def _validate_input(self) -> None:
        """Validate that run directory and log file exist."""
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        
        if not self.log_file.exists():
            available = sorted(self.run_dir.glob("*.log"))
            available_names = ", ".join(p.name for p in available) if available else "none"
            raise FileNotFoundError(
                f"Train log not found: {self.log_file}. Available .log files: {available_names}"
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

    def _load_hungarian_snapshots(self) -> Dict[int, Dict]:
        """Load progression + final snapshots for Hungarian analysis."""
        snapshots_by_epoch: Dict[int, Dict] = {}

        progression_path = self._find_first_existing(["message_progression*.jsonl"])
        final_path = self._find_first_existing(["message_snapshot_final*.json"])

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
        self.generate_summary_report()
        
        print("\n✓ All metrics generated successfully!")
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
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("SURVIVAL GAME - METRICS GENERATOR")
        print("=" * 70)
        
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
