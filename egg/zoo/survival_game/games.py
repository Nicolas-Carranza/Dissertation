#!/usr/bin/env python3
"""
Survival Game — games.py
==========================

Builds the full EGG game object by composing:
    1. Core Sender / Receiver architectures  (archs.py)
    2. EGG RNN wrappers (Reinforce or Gumbel-Softmax)  (egg.core)
    3. Loss function                           (losses.py)

Returns either a SenderReceiverRnnReinforce (mode='rf') or
SenderReceiverRnnGS (mode='gs') game object ready for training.
"""

import egg.core as core
from egg.zoo.survival_game.archs import Sender, Receiver
from egg.zoo.survival_game.losses import SurvivalLoss


def build_game(opts):
    """
    Construct the full communication game (Reinforce or Gumbel-Softmax).

    Expected opts attributes (set by core.init + custom argparse):
        mode:               str   'rf' or 'gs'                    (default: 'rf')
        sender_hidden:      int   hidden size for Sender MLP + RNN  (default: 128)
        receiver_hidden:    int   hidden size for Receiver MLP + RNN (default: 128)
        sender_embedding:   int   embedding dim for Sender RNN       (default: 32)
        receiver_embedding: int   embedding dim for Receiver RNN     (default: 32)
        sender_cell:        str   RNN cell type: 'rnn', 'gru', 'lstm' (default: 'lstm')
        receiver_cell:      str   RNN cell type                       (default: 'lstm')
        vocab_size:         int   message vocabulary size              (default: 50)
        max_len:            int   maximum message length               (default: 6)
        temperature:        float GS temperature (GS mode only)       (default: 2.0)
        sender_entropy_coeff: float  entropy bonus (RF mode only)    (default: 0.1)
        receiver_entropy_coeff: float  entropy bonus (RF mode only)  (default: 0.0)
        reward_scale:       float  reward scaling in loss              (default: 0.01)

    Returns:
        game: SenderReceiverRnnReinforce or SenderReceiverRnnGS
    """
    # ---- Defaults for optional attributes ----
    mode = getattr(opts, "mode", "rf")
    sender_hidden = getattr(opts, "sender_hidden", 128)
    receiver_hidden = getattr(opts, "receiver_hidden", 128)
    sender_embedding = getattr(opts, "sender_embedding", 32)
    receiver_embedding = getattr(opts, "receiver_embedding", 32)
    sender_cell = getattr(opts, "sender_cell", "lstm")
    receiver_cell = getattr(opts, "receiver_cell", "lstm")
    vocab_size = getattr(opts, "vocab_size", 50)
    max_len = getattr(opts, "max_len", 6)
    temperature = getattr(opts, "temperature", 2.0)
    sender_entropy_coeff = getattr(opts, "sender_entropy_coeff", 0.1)
    receiver_entropy_coeff = getattr(opts, "receiver_entropy_coeff", 0.0)
    reward_scale = getattr(opts, "reward_scale", 0.01)
    recon_weight = getattr(opts, "recon_weight", 1.0)

    # ---- Core architectures ----
    sender = Sender(n_hidden=sender_hidden)
    
    # In GS mode, valid_mask is excluded from MLP features (archs.py)
    # but logit masking is still applied to prevent invalid actions.

    #  Also enable the reconstruction head to force sender communication.
    receiver = Receiver(n_hidden=receiver_hidden, recon_head=(mode == "gs"))

    # ---- Loss ----
    action_entropy_coeff = getattr(opts, "action_entropy_coeff", 0.0)
    action_temperature = getattr(opts, "action_temperature", 1.0)
    reward_normalise = getattr(opts, "reward_normalise", False)
    loss = SurvivalLoss(
        reward_scale=reward_scale, mode=mode,
        vocab_size=vocab_size, recon_weight=recon_weight,
        action_entropy_coeff=action_entropy_coeff,
        action_temperature=action_temperature,
        reward_normalise=reward_normalise,
    )

    # ---- Build mode-specific game ----
    if mode == "gs":
        # Gumbel-Softmax: differentiable message channel
        sender = core.RnnSenderGS(
            sender,
            vocab_size=vocab_size,
            embed_dim=sender_embedding,
            hidden_size=sender_hidden,
            cell=sender_cell,
            max_len=max_len,
            temperature=temperature,
        )
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=vocab_size,
            embed_dim=receiver_embedding,
            hidden_size=receiver_hidden,
            cell=receiver_cell,
        )
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
    else:
        # Reinforce: discrete message channel with policy gradient
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=vocab_size,
            embed_dim=sender_embedding,
            hidden_size=sender_hidden,
            cell=sender_cell,
            max_len=max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=vocab_size,
            embed_dim=receiver_embedding,
            hidden_size=receiver_hidden,
            cell=receiver_cell,
        )
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=sender_entropy_coeff,
            receiver_entropy_coeff=receiver_entropy_coeff,
        )

    return game
