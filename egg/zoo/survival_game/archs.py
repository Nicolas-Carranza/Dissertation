#!/usr/bin/env python3
"""
Survival Game — archs.py
=========================

Core Sender and Receiver agent architectures.

In EGG, the "core" agent is a small neural network that:
  - Sender: maps the input to a hidden state (used to initialise the message RNN)
  - Receiver: maps the RNN-encoded message + game state to an action distribution

These cores are then wrapped by EGG's RnnSenderReinforce / RnnReceiverDeterministic
wrappers in games.py.

Architecture summary:
    Sender:   MLP  (30 → hidden → hidden)   input = one-hot entity vector
    Receiver: MLP  (rnn_hidden + 27 → hidden → 11)   output = action logits
              The 27-dim receiver_input = 16 state + 11 valid-action mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.survival_game.prototype import N_ACTIONS, VALUES_PER_DIM, VECTOR_DIM, ALL_ENTITIES

# Sender input: 6 dimensions × 5 one-hot values = 30
SENDER_INPUT_DIM = VECTOR_DIM * VALUES_PER_DIM

# Receiver input transported from data.py: 16 (game state) + 11 (valid mask) = 27
# BUT only the 16 state dims are used as MLP features.
# The valid mask is extracted solely for logit masking of invalid actions.
RECEIVER_INPUT_DIM = 16 + N_ACTIONS  # 27 (transport format)
RECEIVER_STATE_DIM = 16               # features actually fed to MLP

# Entity-level reconstruction target (40 individual entities)
# This forces the sender to encode WHICH entity it sees, not just what TYPE,
# enabling the receiver to make entity-specific action decisions.
N_ENTITY_TYPES = 5
N_ENTITIES = len(ALL_ENTITIES)  # 40


class Sender(nn.Module):
    """
    Core Sender agent.

    Takes the one-hot entity vector (30 floats) and produces a hidden
    representation that will initialise the message-generating RNN.

    Architecture: Linear → ReLU → Linear → (returned as hidden state)
    """

    def __init__(self, n_hidden: int, n_features: int = SENDER_INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, _aux_input=None):
        """
        Args:
            x: (batch, 30)  one-hot entity vector
        Returns:
            hidden: (batch, n_hidden)  used to init the RNN
        """
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h


class Receiver(nn.Module):
    """
    Core Receiver agent.

    Takes the RNN-encoded message hidden state concatenated with the
    game state + valid action mask, and outputs logits over the 11 actions.

    Architecture: Linear → ReLU → Linear → output (logits)

    Invalid actions are masked to -1e9 before output, so the
    receiver learns to only pick valid actions.
    """

    def __init__(self, n_hidden: int, n_actions: int = N_ACTIONS, apply_mask: bool = True,
                 recon_head: bool = False):
        super().__init__()
        # Input: rnn_hidden (from message) + state only (16 dims)
        # The valid_mask is NOT part of the MLP features, this forces
        # the receiver to rely on the sender's message for entity info.
        self.fc1 = nn.Linear(n_hidden + RECEIVER_STATE_DIM, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.n_actions = n_actions
        self.apply_mask = apply_mask
        self.recon_head = recon_head

        if recon_head:
            # Auxiliary entity-IDENTITY prediction from message hidden state ONLY.
            # This head reads from x (message RNN output) directly, so its
            # gradient flows: CE → fc_recon → x → receiver RNN → message → sender.
            # Predicts which of the 40 entities was seen (not just 5 types),
            # forcing the sender to encode fine-grained entity identity.
            self.fc_recon = nn.Sequential(
                nn.Linear(n_hidden, 64),
                nn.ReLU(),
                nn.Linear(64, N_ENTITIES),
            )

    def forward(self, x, receiver_input=None, _aux_input=None):
        """
        Args:
            x: (batch, rnn_hidden)   the RNN-encoded message
            receiver_input: (batch, 27)   game state (16) + valid mask (11)
        Returns:
            output: (batch, 11) action logits (masked) if no recon_head,
                    (batch, 51) action logits (11) + entity_pred (40) if recon_head
        """
        if receiver_input is not None:
            # Split: first 16 = state features, last 11 = valid-action mask
            state_vec = receiver_input[:, :16]       # (batch, 16)
            valid_mask = receiver_input[:, 16:]       # (batch, 11)

            # MLP sees only message + state — NOT the mask
            combined = torch.cat([x, state_vec], dim=-1)
        else:
            combined = x
            valid_mask = None

        h = F.relu(self.fc1(combined))
        logits = self.fc2(h)

        # Mask invalid actions with large negative value
        if valid_mask is not None and self.apply_mask:
            invalid_mask = (valid_mask == 0.0)
            logits = logits.masked_fill(invalid_mask, -1e9)

        if self.recon_head:
            # Entity-identity prediction from message hidden (x) ONLY
            # x carries the receiver RNN's representation of the sender's message
            entity_pred = self.fc_recon(x)  # (batch, 40)
            return torch.cat([logits, entity_pred], dim=-1)  # (batch, 51)

        return logits
