# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import egg.core as core
from architectures import MNISTSender, MNISTDiscriReceiver
from data_readers_mnist import MNISTDiscriDataset


def get_params(params):
    parser = argparse.ArgumentParser()
    
    # MNIST-specific arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to download/store MNIST data (default: ./data)",
    )
    parser.add_argument(
        "--n_distractors",
        type=int,
        default=2,
        help="Number of distractor images per sample (default: 2, meaning 3 total images)",
    )
    
    # Training method arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-2)",
    )
    
    # Agent architecture arguments
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="gru",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: gru)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="gru",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: gru)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=256,
        help="Size of the hidden layer of Sender (default: 256)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=256,
        help="Size of the hidden layer of Receiver (default: 256)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=50,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 50)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=50,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 50)",
    )
    
    # Output arguments
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints validation events",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Path to save model checkpoint after training (default: None)",
    )
    
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    
    # Load MNIST datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    print("Loading MNIST data...", flush=True)
    train_mnist = datasets.MNIST(
        opts.data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_mnist = datasets.MNIST(
        opts.data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create discrimination game datasets
    train_ds = MNISTDiscriDataset(train_mnist, n_distractors=opts.n_distractors, seed=opts.random_seed)
    test_ds = MNISTDiscriDataset(test_mnist, n_distractors=opts.n_distractors, seed=opts.random_seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for speed
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train dataset size: {len(train_ds)}", flush=True)
    print(f"Test dataset size: {len(test_ds)}", flush=True)
    print(f"Images per sample: {opts.n_distractors + 1} (1 target + {opts.n_distractors} distractors)", flush=True)
    
    # Define loss function (same as discrimination game)
    def loss(
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        # Accuracy: check if highest score matches target position
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        # Cross-entropy loss
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}
    
    # Build core agents
    sender = MNISTSender(n_hidden=opts.sender_hidden)
    receiver = MNISTDiscriReceiver(n_hidden=opts.receiver_hidden)
    
    # Wrap with EGG wrappers
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
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:  # REINFORCE (default)
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
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []
    
    # Build optimizer
    optimizer = core.build_optimizer(game.parameters())
    
    # Create trainer
    if opts.print_validation_events:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks + [
                core.ConsoleLogger(print_train_loss=True, as_json=True)
            ],
        )
    
    # Train!
    print("Starting training...", flush=True)
    trainer.train(n_epochs=opts.n_epochs)
    
    # Save checkpoint if requested
    if opts.save_checkpoint:
        print(f"\nSaving checkpoint to {opts.save_checkpoint}...", flush=True)
        torch.save({
            'sender': sender.state_dict(),
            'receiver': receiver.state_dict(),
            'optimizer': optimizer.state_dict(),
            'opts': vars(opts)
        }, opts.save_checkpoint)
        print("Checkpoint saved!", flush=True)
    
    print("Training complete!", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
