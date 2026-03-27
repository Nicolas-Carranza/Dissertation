# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to analyze emergent communication patterns in trained MNIST discrimination game.
Samples a subset of images per digit class and logs the messages generated.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Ensure imports work when running this script directly by path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import egg.core as core
from architectures import MNISTSender, MNISTDiscriReceiver
from data_readers_mnist import MNISTDiscriDataset


def _unpack_sender_output(sender_output):
    """Support both RF and GS sender wrapper return formats."""
    if isinstance(sender_output, (tuple, list)):
        message = sender_output[0]
        entropy = sender_output[2] if len(sender_output) > 2 else None
        return message, entropy
    return sender_output, None


def _unpack_receiver_output(receiver_output):
    """Support both RF and GS receiver wrapper return formats."""
    if isinstance(receiver_output, (tuple, list)):
        return receiver_output[0]
    return receiver_output


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST data directory")
    parser.add_argument("--n_distractors", type=int, default=2, help="Number of distractors")
    parser.add_argument("--samples_per_digit", type=int, default=100, help="Number of samples to analyze per digit (default: 100)")
    parser.add_argument("--vocab_size", type=int, default=50, help="Vocabulary size")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum message length")
    parser.add_argument("--sender_hidden", type=int, default=256, help="Sender hidden size")
    parser.add_argument("--receiver_hidden", type=int, default=256, help="Receiver hidden size")
    parser.add_argument("--sender_embedding", type=int, default=50, help="Sender embedding size")
    parser.add_argument("--receiver_embedding", type=int, default=50, help="Receiver embedding size")
    parser.add_argument("--sender_cell", type=str, default="gru", help="Sender cell type")
    parser.add_argument("--receiver_cell", type=str, default="gru", help="Receiver cell type")
    parser.add_argument("--mode", type=str, default="rf", choices=["rf", "gs"], help="Checkpoint training mode: rf or gs")
    parser.add_argument("--temperature", type=float, default=1.0, help="GS sender temperature (used when --mode gs)")
    parser.add_argument("--output", type=str, default="message_analysis.json", help="Output file for analysis")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling and evaluation")
    parser.add_argument("--top_k", type=int, default=5, help="Number of most frequent messages to keep per digit")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        default=False,
        help="If set, save only per-digit summary and top-k messages (omit all_messages list)",
    )
    
    return parser.parse_args()


def sample_by_digit(mnist_dataset, samples_per_digit=100):
    """Sample a balanced subset of MNIST with samples_per_digit per class."""
    digit_indices = defaultdict(list)
    
    # Group indices by digit
    for idx in range(len(mnist_dataset)):
        _, label = mnist_dataset[idx]
        digit_indices[label].append(idx)
    
    # Sample from each digit
    sampled_indices = []
    for digit in range(10):
        indices = digit_indices[digit]
        n_samples = min(samples_per_digit, len(indices))
        sampled = np.random.choice(indices, size=n_samples, replace=False)
        sampled_indices.extend(sampled)
    
    return sampled_indices


def analyze_messages(opts):
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    print(f"Loading MNIST data from {opts.data_dir}...", flush=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_mnist = datasets.MNIST(opts.data_dir, train=False, download=True, transform=transform)
    
    # Sample subset for analysis
    print(f"Sampling {opts.samples_per_digit} images per digit class...", flush=True)
    sampled_indices = sample_by_digit(test_mnist, opts.samples_per_digit)
    subset_mnist = Subset(test_mnist, sampled_indices)
    
    test_ds = MNISTDiscriDataset(subset_mnist, n_distractors=opts.n_distractors, seed=opts.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Total samples to analyze: {len(test_ds)}", flush=True)
    
    # Build model
    print("Building model...", flush=True)
    sender = MNISTSender(n_hidden=opts.sender_hidden)
    receiver = MNISTDiscriReceiver(n_hidden=opts.receiver_hidden)
    
    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
    else:
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
    
    # Load checkpoint
    print(f"Loading checkpoint from {opts.checkpoint}...", flush=True)
    checkpoint = torch.load(opts.checkpoint, map_location=opts.device, weights_only=False)
    sender.load_state_dict(checkpoint['sender'])
    receiver.load_state_dict(checkpoint['receiver'])
    
    sender.eval()
    receiver.eval()
    
    # Analyze messages
    print("Analyzing messages...", flush=True)
    results = defaultdict(list)
    
    with torch.no_grad():
        for idx, (sender_input, target_pos, receiver_input) in enumerate(test_loader):
            sender_input = sender_input.to(opts.device)
            receiver_input = receiver_input.to(opts.device)
            target_pos = target_pos.to(opts.device)
            
            # Get the actual digit label from the original dataset
            original_idx = sampled_indices[idx]
            _, digit_label = test_mnist[original_idx]
            
            # Generate message
            sender_output = sender(sender_input, None)
            message, entropy = _unpack_sender_output(sender_output)

            # For GS wrappers, message is a distribution over vocab at each step.
            # Convert to discrete token ids for counting/reporting only.
            if message.dim() == 3:
                message_tokens = message.argmax(dim=-1)
            else:
                message_tokens = message
            
            # Get receiver prediction - receiver returns (output, logits, entropy)
            receiver_output = receiver(message, receiver_input, None)
            output = _unpack_receiver_output(receiver_output)
            prediction = output.argmax(dim=1).item()

            if entropy is not None:
                entropy_value = float(entropy.mean().item())
            elif message.dim() == 3:
                probs = message[0].clamp_min(1e-12)
                token_entropy = -(probs * probs.log()).sum(dim=-1)
                entropy_value = float(token_entropy.mean().item())
            else:
                entropy_value = 0.0

            token_list = message_tokens[0].detach().cpu().tolist()
            
            # Store results
            results[int(digit_label)].append({
                'message': token_list,
                'length': len([x for x in token_list if x > 0]),  # Exclude padding
                'entropy': entropy_value,
                'correct': prediction == target_pos.item(),
                'predicted_pos': prediction,
                'target_pos': target_pos.item()
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(test_ds)} samples...", flush=True)
    
    # Compute statistics per digit
    print("\nComputing statistics...", flush=True)
    summary = {}
    
    for digit in range(10):
        messages_for_digit = results[digit]
        if not messages_for_digit:
            continue
        
        accuracies = [m['correct'] for m in messages_for_digit]
        lengths = [m['length'] for m in messages_for_digit]
        entropies = [m['entropy'] for m in messages_for_digit]
        
        # Find most common messages (as tuples for hashing)
        message_counts = defaultdict(int)
        for m in messages_for_digit:
            msg_tuple = tuple(m['message'][:m['length']])  # Only non-padded part
            message_counts[msg_tuple] += 1
        
        top_messages = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:opts.top_k]
        
        summary_entry = {
            'accuracy': np.mean(accuracies),
            'avg_length': np.mean(lengths),
            'avg_entropy': np.mean(entropies),
            'n_samples': len(messages_for_digit),
            f'top_{opts.top_k}_messages': [
                {'message': list(msg), 'count': count, 'frequency': count/len(messages_for_digit)}
                for msg, count in top_messages
            ],
        }

        if not opts.summary_only:
            summary_entry['all_messages'] = messages_for_digit  # Full data

        summary[digit] = summary_entry
    
    # Save results
    print(f"\nSaving results to {opts.output}...", flush=True)
    with open(opts.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MESSAGE ANALYSIS SUMMARY")
    print("="*60)
    
    for digit in range(10):
        if digit in summary:
            s = summary[digit]
            print(f"\nDigit {digit}:")
            print(f"  Accuracy: {s['accuracy']*100:.1f}%")
            print(f"  Avg message length: {s['avg_length']:.2f}")
            print(f"  Avg entropy: {s['avg_entropy']:.4f}")
            top_key = f"top_{opts.top_k}_messages"
            if s[top_key]:
                print(f"  Most common message: {s[top_key][0]['message']} "
                    f"({s[top_key][0]['frequency']*100:.1f}% of samples)")
    
    print(f"\nFull analysis saved to: {opts.output}")
    print("="*60)


if __name__ == "__main__":
    opts = get_params()
    analyze_messages(opts)
