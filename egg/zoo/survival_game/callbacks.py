#!/usr/bin/env python3
"""
Survival Game — callbacks.py
==============================

Custom EGG callbacks for monitoring survival-game-specific metrics.

SurvivalGameEvaluator:
    At the end of each epoch (or every N epochs), runs full episodes using
    the trained Sender + Receiver agents and reports:
        - Survival rate (% of episodes where agents survive all turns)
        - Mean reward per episode
        - Action distribution
        - Mean message length
        - Communication gap (trained agents vs. random baseline)

This provides a more realistic evaluation than per-turn loss alone,
since the per-turn loss uses randomly-generated game states while
the evaluator runs proper sequential episodes.
"""

import random
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from egg.core import Callback, Interaction

from egg.zoo.survival_game.prototype import (
    ALL_ENTITIES,
    ACTION_RESOLVERS,
    ACTION_NAMES,
    N_ACTIONS,
    VECTOR_DIM,
    VALUES_PER_DIM,
    ANIMAL,
    RESOURCE,
    DANGER,
    CRAFT_OPP,
    EVENT,
    HUNT,
    GATHER,
    FLEE,
    REST,
    MITIGATE,
    ENDURE,
    EAT,
    CRAFT_SPEAR,
    CRAFT_FIRE,
    CRAFT_SHELTER,
    CRAFT_ROD,
    Entity,
    GameState,
    Inventory,
    generate_encounter,
    get_valid_actions,
    update_weather,
    _apply_unaddressed_danger,
)
from egg.zoo.survival_game.data import entity_to_onehot, game_state_to_tensor


class SurvivalGameEvaluator(Callback):
    """
    Runs full multi-turn episodes using the trained agents at the end of
    selected epochs to measure true survival rate and communication quality.

    The evaluation uses the full game loop:
        1. Generate encounter → Sender sees entity → produces message
        2. Receiver sees game state + valid mask + message → picks action
        3. Action resolved, state updates, repeat for max_turns
    """

    def __init__(
        self,
        n_episodes: int = 100,
        max_turns: int = 20,
        eval_freq: int = 5,
        verbose: bool = False,
    ):
        """
        Args:
            n_episodes: number of episodes to run per evaluation
            max_turns:  turns per episode
            eval_freq:  evaluate every N epochs (0 = never)
            verbose:    print per-episode details
        """
        super().__init__()
        self.n_episodes = n_episodes
        self.max_turns = max_turns
        self.eval_freq = eval_freq
        self.verbose = verbose

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.eval_freq <= 0 or epoch % self.eval_freq != 0:
            return

        game = self.trainer.game
        game.eval()

        results = self._run_evaluation(game)

        # Print results
        print(f"\n{'='*60}", flush=True)
        print(f"  Survival Game Evaluation — Epoch {epoch}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Episodes:      {self.n_episodes}", flush=True)
        print(f"  Survival rate: {results['survival_rate']:.1%}", flush=True)
        print(f"  Mean reward:   {results['mean_reward']:.1f}", flush=True)
        print(f"  Mean msg len:  {results['mean_msg_len']:.2f}", flush=True)
        print(f"  Valid action%: {results['valid_rate']:.1%}", flush=True)

        # Action distribution
        print(f"  Action distribution:", flush=True)
        for action_id in range(N_ACTIONS):
            name = ACTION_NAMES.get(action_id, "?")
            rate = results['action_dist'].get(action_id, 0.0)
            bar = "█" * int(rate * 40)
            print(f"    {name:>14}: {rate:5.1%} {bar}", flush=True)
        print(f"{'='*60}\n", flush=True)

        game.train()

    @torch.no_grad()
    def _run_evaluation(self, game) -> dict:
        """
        Run full episodes using the trained game's sender and receiver.
        Supports both Reinforce and Gumbel-Softmax modes.
        """
        from egg.core.gs_wrappers import RnnSenderGS

        sender = game.sender
        receiver = game.receiver
        device = next(game.parameters()).device
        is_gs = isinstance(sender, RnnSenderGS)

        survived = 0
        total_reward = 0.0
        total_turns = 0
        total_valid = 0
        action_counts = defaultdict(int)
        msg_lengths = []

        for ep in range(self.n_episodes):
            state = GameState(max_turns=self.max_turns)
            ep_reward = 0.0

            for turn in range(1, self.max_turns + 1):
                state.turn = turn

                # Metabolic drain
                drain = random.randint(2, 4)
                state.energy -= drain

                # Weather update
                update_weather(state)

                # Generate encounter
                target, distractors = generate_encounter(state, n_distractors=2)

                # Get valid actions
                valid = get_valid_actions(target, state)

                # ---- Sender input ----
                sender_input = entity_to_onehot(target.vector).unsqueeze(0).to(device)

                # ---- Receiver input ----
                state_vec = game_state_to_tensor(state).unsqueeze(0).to(device)
                valid_mask = torch.zeros(1, N_ACTIONS, device=device)
                for a in valid:
                    valid_mask[0, a] = 1.0
                receiver_input = torch.cat([state_vec, valid_mask], dim=-1)

                # ---- Get message + receiver output (mode-dependent) ----
                if is_gs:
                    # GS: sender returns (1, max_len+1, vocab_size) hard one-hot in eval
                    message = sender(sender_input)
                    # Receiver returns (1, max_len+1, n_actions)
                    receiver_outputs = receiver(message, receiver_input)

                    # Find first EOS step (vocab index 0 = EOS)
                    eos_mask = message[0, :, 0]  # (max_len+1,)
                    eos_steps = eos_mask.nonzero(as_tuple=True)[0]
                    if len(eos_steps) > 0:
                        eos_step = eos_steps[0].item()
                    else:
                        eos_step = message.size(1) - 1
                    receiver_output = receiver_outputs[:, eos_step, :]
                    msg_lengths.append(eos_step + 1)
                else:
                    # RF: sender returns (message, log_prob, entropy)
                    message, log_prob_s, entropy_s = sender(sender_input)
                    from egg.core.reinforce_wrappers import find_lengths
                    message_length = find_lengths(message)
                    receiver_output, log_prob_r, entropy_r = receiver(
                        message, receiver_input, None, message_length
                    )
                    msg_lengths.append(message_length.item())

                # ---- Pick action ----
                # In GS mode, receiver_output has 16 dims (11 action + 5 recon);
                # take argmax over the first 11 action logits only.
                action_logits = receiver_output[:, :11] if is_gs else receiver_output
                action_id = int(action_logits.argmax(dim=-1).item())
                action_counts[action_id] += 1
                total_turns += 1

                if action_id in valid:
                    total_valid += 1

                # ---- Resolve action ----
                resolver = ACTION_RESOLVERS.get(action_id)
                if resolver is None or action_id not in valid:
                    reward = -15.0
                    desc = "INVALID"
                else:
                    reward, desc = resolver(target, state)

                # Unaddressed encounter effects
                addressed = (
                    (action_id == HUNT and target.entity_type == ANIMAL) or
                    (action_id == GATHER and target.entity_type in (RESOURCE, CRAFT_OPP, EVENT)) or
                    (action_id == MITIGATE and target.entity_type == DANGER) or
                    (action_id == ENDURE and target.entity_type == DANGER) or
                    (action_id == FLEE)
                )
                if not addressed:
                    _apply_unaddressed_danger(target, state)

                # Alive bonus
                reward += 10.0
                if state.energy > 70:
                    reward += 5.0
                if state.health > 70:
                    reward += 3.0
                ep_reward += reward

                # Clamp
                state.energy = min(max(state.energy, 0.0), 100.0)
                state.health = min(max(state.health, 0.0), 100.0)

                # Death check
                if state.energy <= 0 or state.health <= 0:
                    ep_reward -= 50.0 if state.energy <= 0 else 300.0
                    break

            if state.energy > 0 and state.health > 0:
                survived += 1
            total_reward += ep_reward

        # Aggregate
        n = float(self.n_episodes)
        action_dist = {a: action_counts[a] / max(total_turns, 1)
                       for a in range(N_ACTIONS)}

        return {
            "survival_rate": survived / n,
            "mean_reward": total_reward / n,
            "mean_msg_len": sum(msg_lengths) / max(len(msg_lengths), 1),
            "valid_rate": total_valid / max(total_turns, 1),
            "action_dist": action_dist,
        }


class MessageAnalyzer(Callback):
    """
    Periodically analyzes the emergent language by looking at which messages
    are sent for different entity types during validation.

    Prints a simple table showing most common messages per entity type.
    """

    def __init__(self, analyze_freq: int = 10):
        super().__init__()
        self.analyze_freq = analyze_freq

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.analyze_freq <= 0 or epoch % self.analyze_freq != 0:
            return

        if logs.message is None or logs.sender_input is None:
            return

        messages = logs.message  # RF: (N, max_len), GS: (N, max_len+1, vocab_size)

        # GS mode: messages are soft distributions — convert to discrete
        if messages.dim() == 3:
            messages = messages.argmax(dim=-1)  # (N, max_len+1)
        sender_inputs = logs.sender_input  # (N, 30)

        # Decode entity types from sender_input (one-hot: first 5 values = entity_type)
        # entity_type is in dims 0-4 of the 30-dim one-hot vector
        entity_types = sender_inputs[:, :VALUES_PER_DIM].argmax(dim=-1)  # (N,)

        # Group messages by entity type
        type_names = {0: "Animal", 1: "Resource", 2: "Danger",
                      3: "CraftOpp", 4: "Event"}

        print(f"\n  Message Analysis — Epoch {epoch}", flush=True)
        print(f"  {'Entity Type':>12} | {'Count':>6} | {'Unique':>6} | {'Top messages':>40}", flush=True)
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*40}", flush=True)

        for etype in range(5):
            mask = (entity_types == etype)
            count = int(mask.sum().item())
            if count == 0:
                continue

            etype_msgs = messages[mask]
            # Convert to tuples for counting
            msg_strings = []
            for m in etype_msgs:
                msg_str = " ".join(str(s.item()) for s in m)
                msg_strings.append(msg_str)

            # Count unique messages
            msg_counts = defaultdict(int)
            for ms in msg_strings:
                msg_counts[ms] += 1

            n_unique = len(msg_counts)

            # Top 3
            top = sorted(msg_counts.items(), key=lambda x: -x[1])[:3]
            top_str = " | ".join(f"[{m}]×{c}" for m, c in top)

            tname = type_names.get(etype, f"Type{etype}")
            print(f"  {tname:>12} | {count:>6} | {n_unique:>6} | {top_str}", flush=True)

        print(flush=True)
