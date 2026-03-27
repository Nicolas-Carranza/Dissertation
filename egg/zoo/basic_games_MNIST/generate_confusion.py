"""
Generate confusion matrix (Actual vs Predicted digits) from a trained MNIST discrimination model.

Outputs:
 - `confusion_counts.png` : confusion matrix with raw counts (rows = actual, cols = predicted)
 - `confusion_accuracy.png`: same matrix but colored by row-normalized accuracy (proportion)

Usage:
python generate_confusion.py --checkpoint ./checkpoints/working_model.pth --samples_per_digit 1200 --output_prefix confusion

Notes:
- This script recreates the distractor sampling and shuffling used in `analyze_messages.py` (random seed = 42 by default).
- It will analyze up to `samples_per_digit` per class, capped by availability.
- Requires: torch, torchvision, matplotlib. seaborn is optional (used if available).
"""

import argparse
import random
import math
import sys
from pathlib import Path

import torch
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# Ensure imports work when running this script directly by path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architectures import MNISTSender, MNISTDiscriReceiver
import egg.core as core


def _unpack_sender_output(sender_output):
    if isinstance(sender_output, (tuple, list)):
        return sender_output[0]
    return sender_output


def _unpack_receiver_output(receiver_output):
    if isinstance(receiver_output, (tuple, list)):
        return receiver_output[0]
    return receiver_output


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--samples_per_digit", type=int, default=1200)
    p.add_argument("--vocab_size", type=int, default=50)
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--n_distractors", type=int, default=2)
    p.add_argument("--mode", type=str, default="rf", choices=["rf", "gs"])
    p.add_argument("--temperature", type=float, default=1.0, help="GS sender temperature (used when --mode gs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_prefix", type=str, default="confusion")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def sample_by_digit_indices(mnist_dataset, samples_per_digit=100, seed=42):
    from collections import defaultdict
    digit_indices = defaultdict(list)
    for idx in range(len(mnist_dataset)):
        _, label = mnist_dataset[idx]
        digit_indices[int(label)].append(idx)

    sampled_indices = []
    rng = np.random.RandomState(seed)
    for digit in range(10):
        indices = digit_indices[digit]
        n = min(samples_per_digit, len(indices))
        sampled = list(rng.choice(indices, size=n, replace=False))
        sampled_indices.extend(sampled)
    return sampled_indices


def build_model(opts):
    sender = MNISTSender(n_hidden=256)
    receiver = MNISTDiscriReceiver(n_hidden=256)

    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=50,
            hidden_size=256,
            cell='gru',
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=50,
            hidden_size=256,
            cell='gru',
        )
    else:
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=50,
            hidden_size=256,
            cell='gru',
            max_len=opts.max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=50,
            hidden_size=256,
            cell='gru',
        )

    checkpoint = torch.load(opts.checkpoint, map_location=opts.device, weights_only=False)
    sender.load_state_dict(checkpoint['sender'])
    receiver.load_state_dict(checkpoint['receiver'])
    sender.eval(); receiver.eval()
    return sender, receiver


def main():
    opts = get_args()
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_mnist = datasets.MNIST(opts.data_dir, train=False, download=True, transform=transform)

    sampled_indices = sample_by_digit_indices(test_mnist, opts.samples_per_digit, seed=opts.seed)
    print(f"Total samples to analyze: {len(sampled_indices)}")

    sender, receiver = build_model(opts)

    # confusion: rows actual digit (0..9), cols predicted digit (0..9)
    confusion = np.zeros((10,10), dtype=np.int64)

    # For each sampled target index, create distractors and shuffled images (same logic as dataset)
    for i, target_idx in enumerate(sampled_indices):
        target_img, target_label = test_mnist[target_idx]
        # pick distractors (ensure different images; they may have same digit but that's fine)
        distractor_indices = random.sample(range(len(test_mnist)), opts.n_distractors)
        all_indices = [target_idx] + distractor_indices
        positions = list(range(len(all_indices)))
        random.shuffle(positions)
        # reorder indices according to shuffle
        shuffled_indices = [all_indices[p] for p in positions]

        # find where target ended up
        target_pos = positions.index(0)

        # build sender input (target image)
        sender_input = target_img.view(-1).unsqueeze(0)  # (1,784)
        # build receiver_input images tensor
        imgs = [test_mnist[idx][0] for idx in shuffled_indices]
        receiver_input = torch.stack([img.view(-1) for img in imgs]).unsqueeze(0)  # (1, n_images, 784)

        # run through models
        with torch.no_grad():
            sender_output = sender(sender_input, None)
            message = _unpack_sender_output(sender_output)
            receiver_output = receiver(message, receiver_input, None)
            out = _unpack_receiver_output(receiver_output)
            pred_pos = int(out.argmax(dim=1).item())

        # map pred_pos to predicted digit label using shuffled_indices
        pred_idx = shuffled_indices[pred_pos]
        pred_digit = int(test_mnist[pred_idx][1])
        true_digit = int(target_label)

        confusion[true_digit, pred_digit] += 1

        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(sampled_indices)}")

    # Save confusion arrays and plot
    np.save(opts.output_prefix + "_counts.npy", confusion)

    # Row-normalized (per-actual) accuracy proportions
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_norm = confusion.astype(float) / np.where(row_sums==0, 1, row_sums)

    # Plot counts with annotations
    plt.figure(figsize=(10,8))
    if _HAS_SEABORN:
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={'label': 'Number of samples'})
    else:
        plt.imshow(confusion, interpolation='nearest', cmap='Blues')
        cbar = plt.colorbar()
        cbar.set_label('Number of samples', rotation=270, labelpad=20)
        for (r,c), val in np.ndenumerate(confusion):
            plt.text(c, r, str(val), ha='center', va='center', color='black', fontsize=8)
        plt.xticks(range(10))
        plt.yticks(range(10))
    plt.xlabel('Predicted digit')
    plt.ylabel('Actual digit')
    plt.title('Confusion matrix (counts)')
    plt.savefig(opts.output_prefix + '_counts.png', dpi=200, bbox_inches='tight')
    print(f"Saved {opts.output_prefix}_counts.png")

    # Plot normalized heatmap (accuracy proportions)
    plt.figure(figsize=(10,8))
    if _HAS_SEABORN:
        sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='viridis', cbar=True,
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={'label': 'Proportion (0-1)'})
    else:
        plt.imshow(confusion_norm, interpolation='nearest', cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Proportion (0-1)', rotation=270, labelpad=20)
        for (r,c), val in np.ndenumerate(confusion_norm):
            plt.text(c, r, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)
        plt.xticks(range(10))
        plt.yticks(range(10))
    plt.xlabel('Predicted digit')
    plt.ylabel('Actual digit')
    plt.title('Confusion matrix (row-normalized proportions)')
    plt.savefig(opts.output_prefix + '_accuracy.png', dpi=200, bbox_inches='tight')
    print(f"Saved {opts.output_prefix}_accuracy.png")

if __name__ == '__main__':
    main()
