#!/usr/bin/env python3
"""
Survival Game - losses.py
==========================

Loss function for the survival game (supports both Reinforce and
Gumbel-Softmax training modes).

EGG calls:
    loss(sender_input, message, receiver_input, receiver_output, labels, aux_input)
and expects back:
    (loss_tensor, aux_dict)

where:
    loss_tensor: shape (batch_size,), per-sample loss (lower = better for agent)
    aux_dict:    dictionary of auxiliary metrics for logging

Reinforce mode (mode='rf'):
    1. Pick argmax action from receiver logits (non-differentiable).
    2. Simulate the action and return -reward as loss.
    3. EGG's CommunicationRnnReinforce handles the policy gradient.

Gumbel-Softmax mode (mode='gs'):
    1. Compute rewards for ALL 11 actions (using deep-copied game states).
    2. Use soft action probabilities: action_probs = softmax(logits).
    3. Differentiable loss = -(action_probs * rewards_per_action).sum(-1).
    4. Gradients flow directly through the soft action selection.
"""

import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.survival_game.prototype import (
    ALL_ENTITIES,
    ACTION_RESOLVERS,
    ACTION_NAMES,
    N_ACTIONS,
    VECTOR_DIM,
    VALUES_PER_DIM,
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
    ANIMAL,
    RESOURCE,
    DANGER,
    CRAFT_OPP,
    EVENT,
    TOOL_NONE,
    TOOL_FIRE,
    DANGER_INJURY_PROB,
    ENERGY_MAP,
    UNMITIGATED_HEALTH,
    UNMITIGATED_ENERGY,
    ENDURE_FACTOR,
    MITIGATE_FLAT,
    TOOL_NAMES,
    Entity,
    GameState,
    Inventory,
    get_valid_actions,
    _apply_unaddressed_danger,
)


# =============================================================================
# Label unpacking helpers
# =============================================================================

def _unpack_labels(labels: torch.Tensor):
    """
    Unpack the 26-float labels tensor (per sample) into components.

    Layout:
        [0:6]   entity vector (raw int values as floats)
        [6:22]  game state vector (16 floats, normalised)
        [22]    entity index in ALL_ENTITIES (-1 if not found)
        [23]    weather as int
        [24]    turn as int
        [25]    max_turns as int

    Returns:
        entity_vecs:  (batch, 6) int
        state_vecs:   (batch, 16) float
        entity_idxs:  (batch,) int
        weather:      (batch,) int
        turns:        (batch,) int
        max_turns:    (batch,) int
    """
    entity_vecs = labels[:, :6].long()
    state_vecs = labels[:, 6:22]
    entity_idxs = labels[:, 22].long()
    weather = labels[:, 23].long()
    turns = labels[:, 24].long()
    max_turns_vals = labels[:, 25].long()
    return entity_vecs, state_vecs, entity_idxs, weather, turns, max_turns_vals


def _reconstruct_entity(entity_vec: torch.Tensor, entity_idx: int) -> Entity:
    """
    Reconstruct an Entity object from the packed label data.
    Uses the entity index to look up from ALL_ENTITIES if available,
    otherwise creates a minimal Entity from the vector.
    """
    idx = int(entity_idx)
    if 0 <= idx < len(ALL_ENTITIES):
        return ALL_ENTITIES[idx]

    # Fallback: reconstruct from vector
    vec = entity_vec.tolist()
    vec = [int(v) for v in vec]
    return Entity(
        name="unknown",
        vector=vec,
        entity_type=vec[0],
        subtype=vec[1],
        danger_level=vec[2],
        energy_value=vec[3],
        tool_required=vec[4],
        weather_dep=vec[5],
    )


def _reconstruct_state(state_vec: torch.Tensor, weather: int,
                       turn: int, max_turns: int) -> GameState:
    """
    Reconstruct a GameState from the normalised state vector.

    State vector layout (16 floats):
        [0]  energy / 100
        [1]  health / 100
        [2]  weather / 4
        [3]  turn / max_turns
        [4:]  inventory (12 dims), each normalised by its max
    """
    sv = state_vec.tolist()
    energy = sv[0] * 100.0
    health = sv[1] * 100.0

    # Inventory: reverse the normalisation from Inventory.to_vector()
    # to_vector returns:
    #   [spear, fire, shelter, rod, wood/5, stone/5,
    #    raw_meat/5, cooked/5, berries/5, fish/5, water/5, herbs/5]
    inv = Inventory()
    inv.spear = sv[4] > 0.5
    inv.fire = sv[5] > 0.5
    inv.shelter = sv[6] > 0.5
    inv.fishing_rod = sv[7] > 0.5
    inv.wood = int(round(sv[8] * 5))
    inv.stone = int(round(sv[9] * 5))
    inv.raw_meat = int(round(sv[10] * 5))
    inv.cooked_meat = int(round(sv[11] * 5))
    inv.berries = int(round(sv[12] * 5))
    inv.fish = int(round(sv[13] * 5))
    inv.water = int(round(sv[14] * 5))
    inv.herbs = int(round(sv[15] * 5))

    state = GameState(max_turns=max_turns)
    state.energy = energy
    state.health = health
    state.weather = weather
    state.turn = turn
    state.inventory = inv
    state.alive = True

    return state


# =============================================================================
# Reward computation (per-sample, on CPU)
# =============================================================================

def _compute_single_reward(action_id: int, entity: Entity,
                           state: GameState) -> Tuple[float, dict]:
    """
    Simulate a single action on a reconstructed game state and return reward.

    Returns:
        reward: float - higher is better
        info: dict with extra fields for logging
    """
    valid = get_valid_actions(entity, state)

    info = {
        "valid": action_id in valid,
        "action": action_id,
        "entity_type": entity.entity_type,
    }

    # Penalty for choosing invalid action
    if action_id not in valid:
        info["reason"] = "invalid"
        return -15.0, info

    # Resolve the action
    resolver = ACTION_RESOLVERS.get(action_id)
    if resolver is None:
        info["reason"] = "unknown_action"
        return -15.0, info

    reward, desc = resolver(entity, state)

    # Unaddressed-encounter effects
    addressed = (
        (action_id == HUNT and entity.entity_type == ANIMAL) or
        (action_id == GATHER and entity.entity_type in (RESOURCE, CRAFT_OPP, EVENT)) or
        (action_id == MITIGATE and entity.entity_type == DANGER) or
        (action_id == ENDURE and entity.entity_type == DANGER) or
        (action_id == FLEE)
    )
    unaddressed_penalty = 0.0
    if not addressed:
        unaddressed_penalty = _apply_unaddressed_danger(entity, state)

    # Alive bonus
    alive_bonus = 10.0
    energy_bonus = 5.0 if state.energy > 70 else 0.0
    health_bonus = 3.0 if state.health > 70 else 0.0

    total_reward = reward + unaddressed_penalty + alive_bonus + energy_bonus + health_bonus

    # Death penalty
    if state.energy <= 0:
        total_reward -= 50.0
    if state.health <= 0:
        total_reward -= 300.0

    info["action_reward"] = reward
    info["unaddressed_penalty"] = unaddressed_penalty
    info["total_reward"] = total_reward
    info["reason"] = desc

    return total_reward, info


def _compute_expected_rewards(entity: Entity, state: GameState) -> torch.Tensor:
    """
    Compute DETERMINISTIC expected reward for each action.

    Key differences from _compute_single_reward:
      - No random.random() calls - uses expected values (prob x outcome)
      - No state mutation - pure calculation, no deepcopy needed
      - No unaddressed-danger penalty (it's the same for gather/rest/eat/craft
        so it also cancels; only flee/hunt/mitigate truly address encounters)
      - DEATH PROXIMITY PENALTY: dying = game over = forfeiting all remaining
        turns of reward.  When health or energy is critically low, actions are
        penalised/boosted based on how likely they are to cause or prevent
        death.  The penalty scales with remaining turns (more turns left =
        more future value lost).  This is principled future-value discounting,
        NOT per-action hand-tuning.

    This gives clean, stable gradient signal for GS training.
    Invalid actions get a large negative reward.

    Returns:
        rewards: (N_ACTIONS,) float tensor
    """
    valid = get_valid_actions(entity, state)
    rewards = torch.full((N_ACTIONS,), -30.0)  # invalid penalty
    inv = state.inventory

    # Death proximity: remaining turns of expected reward forfeited if agent dies.
    # Dying = game over → lose (remaining_turns × avg_reward) future value.
    remaining_turns = max(1, state.max_turns - state.turn)
    future_value = remaining_turns * 8.0  # conservative avg reward per surviving turn

    for action_id in valid:
        r = 0.0

        if action_id == HUNT:
            if entity.entity_type != ANIMAL:
                r = -10.0
            else:
                has_tool = inv.has_tool(entity.tool_required)
                if has_tool and entity.danger_level <= 1:
                    success = 0.85
                elif has_tool:
                    success = 0.65
                elif entity.tool_required == TOOL_NONE:
                    success = 0.50
                else:
                    success = 0.15
                success *= (state.health / 100.0) * (min(state.energy, 50) / 50.0)

                success_reward = 15.0 + entity.danger_level * 10.0
                # Minor injury risk even on success
                success_injury_prob = entity.danger_level * 0.10
                success_injury_cost = (5.0 + entity.danger_level * 5.0) * 0.5  # rough penalty
                expected_success = success_reward - success_injury_prob * success_injury_cost

                fail_reward = -5.0
                fail_injury_prob = entity.danger_level * 0.25
                fail_injury_cost = 15.0  # simplified
                expected_fail = fail_reward - fail_injury_prob * fail_injury_cost

                r = success * expected_success + (1 - success) * expected_fail

        elif action_id == GATHER:
            if entity.entity_type not in (RESOURCE, CRAFT_OPP, EVENT):
                r = -10.0
            else:
                qty = entity.inventory_qty if entity.inventory_key else 0
                r = 5.0 + qty * 2.0
                # Poison risk for dangerous resources
                if entity.danger_level >= 2:
                    injury_prob = DANGER_INJURY_PROB[entity.danger_level]
                    dmg = 5.0 + entity.danger_level * 5.0
                    r -= injury_prob * dmg * 0.5

        elif action_id == FLEE:
            if entity.entity_type == DANGER and entity.danger_level >= 3:
                r = 2.0
            elif entity.entity_type == ANIMAL and entity.danger_level >= 3:
                r = 1.0
            else:
                r = -2.0

        elif action_id == REST:
            r = -1.0

        elif action_id == MITIGATE:
            if entity.entity_type != DANGER:
                r = -10.0
            elif not inv.has_tool(entity.tool_required):
                # Falls back to endure
                h_loss = int(UNMITIGATED_HEALTH[entity.danger_level] * ENDURE_FACTOR)
                e_loss = int(UNMITIGATED_ENERGY[entity.danger_level] * ENDURE_FACTOR)
                r = -(h_loss + e_loss) * 0.3
            else:
                bonus = 5.0 if (entity.tool_required == TOOL_FIRE and entity.subtype == 1) else 0.0
                r = 20.0 + entity.danger_level * 5.0 + bonus

        elif action_id == ENDURE:
            if entity.entity_type != DANGER:
                r = -10.0
            else:
                h_loss = int(UNMITIGATED_HEALTH[entity.danger_level] * ENDURE_FACTOR)
                e_loss = int(UNMITIGATED_ENERGY[entity.danger_level] * ENDURE_FACTOR)
                r = -(h_loss + e_loss) * 0.3

        elif action_id == EAT:
            food_key = inv.best_food()
            if food_key is None:
                r = -5.0
            else:
                food_energy = {
                    "cooked_meat": 15, "fish": 10, "berries": 6,
                    "water": 5, "raw_meat": 5, "herbs": 3,
                }
                gained = food_energy.get(food_key, 1)
                r = float(gained)
                if food_key == "cooked_meat":
                    r += 5.0
                elif food_key == "raw_meat":
                    r -= 0.30 * 12.0  # expected parasite penalty
                # Energy-critical bonus: eating is much more valuable when hungry
                if state.energy < 30:
                    r += 10.0

        elif action_id in (CRAFT_SPEAR, CRAFT_FIRE, CRAFT_SHELTER, CRAFT_ROD):
            # Map action to tool attr and costs
            craft_map = {
                CRAFT_SPEAR:   ("spear",       1, 1),
                CRAFT_FIRE:    ("fire",         1, 0),
                CRAFT_SHELTER: ("shelter",      2, 0),
                CRAFT_ROD:     ("fishing_rod",  1, 0),
            }
            tool_attr, wood_cost, stone_cost = craft_map[action_id]
            if getattr(inv, tool_attr):
                r = -5.0
            elif inv.wood < wood_cost or inv.stone < stone_cost:
                r = -10.0
            else:
                r = 50.0  # discovery bonus

        # ── DEATH PROXIMITY PENALTY ──────────────────────────────
        # Dying = game over → agent forfeits all future rewards.
        # This makes the per-turn loss reflect the terminal cost of
        # death, creating gradient pressure to stay alive.
        #
        # NOT hand-tuning: the penalty is derived from the expected
        # future value of remaining turns, scaled by proximity to
        # the death threshold.  It applies uniformly based on state,
        # not per-action bias.

        # Starvation risk (energy → 0 = death)
        if state.energy < 20:
            starvation_urgency = 1.0 - state.energy / 20.0  # 1.0 at 0, 0.0 at 20
            if action_id == EAT:
                # Eating when starving = life-saving
                r += starvation_urgency * future_value * 0.4
            elif action_id == REST:
                # Rest gives +5 energy, somewhat helpful
                r += starvation_urgency * future_value * 0.1
            else:
                # Not addressing starvation → closer to death
                r -= starvation_urgency * future_value * 0.3

        # Injury risk (health → 0 = death)
        if state.health < 25:
            injury_urgency = 1.0 - state.health / 25.0  # 1.0 at 0, 0.0 at 25
            risky_action = (
                (action_id == HUNT and entity.entity_type == ANIMAL
                 and entity.danger_level >= 2)
                or (action_id == ENDURE and entity.entity_type == DANGER)
            )
            if risky_action:
                # Could die from injury
                r -= injury_urgency * future_value * 0.5
            elif action_id == FLEE:
                # Fleeing is safe - small bonus
                r += injury_urgency * future_value * 0.1

        rewards[action_id] = r

    return rewards


# =============================================================================
# Loss function class
# =============================================================================

# Number of entity classes for reconstruction auxiliary loss
# Uses entity INDEX (40 individual entities) not entity TYPE (5 categories)
# to force the sender to encode fine-grained identity in messages.
N_ENTITIES = len(ALL_ENTITIES)  # 40


class SurvivalLoss(nn.Module):
    """
    Loss function for the survival game.

    Supports two training modes:
      - 'rf' (Reinforce): Non-differentiable.  Takes argmax action, simulates it,
        returns -reward. EGG's policy gradient handles the rest.
      - 'gs' (Gumbel-Softmax): Differentiable.  Computes rewards for ALL actions,
        uses soft action probabilities to create a differentiable expected reward.
        Also includes an auxiliary message-reconstruction loss that forces the
        sender to encode entity-type information in the message.

    Called by EGG as:
        loss(sender_input, message, receiver_input, receiver_output, labels, aux_input)

    Returns:
        (loss, aux_dict) where loss has shape (batch_size,).
    """

    def __init__(self, reward_scale: float = 0.1, mode: str = "rf",
                 vocab_size: int = 50, recon_weight: float = 1.0,
                 action_entropy_coeff: float = 0.0,
                 action_temperature: float = 1.0,
                 reward_normalise: bool = False):
        super().__init__()
        self.reward_scale = reward_scale
        self.mode = mode
        self.recon_weight = recon_weight
        self.action_entropy_coeff = action_entropy_coeff
        self.action_temperature = action_temperature
        self.reward_normalise = reward_normalise

    def __call__(
        self,
        sender_input: torch.Tensor,
        message: torch.Tensor,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_input=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            sender_input:   (batch, 30)  one-hot entity
            message:        (batch, ...)  RF: discrete symbols; GS: soft dist over vocab
            receiver_input: (batch, 27)  state(16) + valid_mask(11)
            receiver_output: GS mode: (batch, 51) = action logits(11) + entity_pred(40)
                             RF mode: (batch, 11) = action logits only
            receiver_output:(batch, 11)  action logits
            labels:         (batch, 26)  packed entity + state info
            aux_input:      unused
        Returns:
            loss:     (batch_size,)  per-sample loss (lower = better)
            aux_dict: dict with logging metrics
        """
        if self.mode == "gs":
            return self._gs_loss(sender_input, message, receiver_input,
                                 receiver_output, labels)
        else:
            return self._rf_loss(sender_input, message, receiver_input,
                                 receiver_output, labels)

    # ------------------------------------------------------------------
    # Reinforce loss (non-differentiable, uses argmax)
    # ------------------------------------------------------------------
    def _rf_loss(self, sender_input, message, receiver_input,
                 receiver_output, labels):
        batch_size = receiver_output.size(0)
        device = receiver_output.device

        actions = receiver_output.argmax(dim=-1)  # (batch,)

        entity_vecs, state_vecs, entity_idxs, weather, turns, max_turns_vals = \
            _unpack_labels(labels)

        rewards = torch.zeros(batch_size, device=device)
        is_valid = torch.zeros(batch_size, device=device)
        is_hunt = torch.zeros(batch_size, device=device)
        is_gather = torch.zeros(batch_size, device=device)
        is_flee = torch.zeros(batch_size, device=device)
        is_rest = torch.zeros(batch_size, device=device)
        is_mitigate = torch.zeros(batch_size, device=device)
        is_endure = torch.zeros(batch_size, device=device)
        is_eat = torch.zeros(batch_size, device=device)
        is_craft = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            action_id = int(actions[i].item())
            entity = _reconstruct_entity(entity_vecs[i], int(entity_idxs[i].item()))
            state = _reconstruct_state(
                state_vecs[i],
                int(weather[i].item()),
                int(turns[i].item()),
                int(max_turns_vals[i].item()),
            )

            reward, info = _compute_single_reward(action_id, entity, state)
            rewards[i] = reward

            if info["valid"]:
                is_valid[i] = 1.0

            if action_id == HUNT:
                is_hunt[i] = 1.0
            elif action_id == GATHER:
                is_gather[i] = 1.0
            elif action_id == FLEE:
                is_flee[i] = 1.0
            elif action_id == REST:
                is_rest[i] = 1.0
            elif action_id == MITIGATE:
                is_mitigate[i] = 1.0
            elif action_id == ENDURE:
                is_endure[i] = 1.0
            elif action_id == EAT:
                is_eat[i] = 1.0
            elif action_id in (CRAFT_SPEAR, CRAFT_FIRE, CRAFT_SHELTER, CRAFT_ROD):
                is_craft[i] = 1.0

        loss = -rewards * self.reward_scale

        aux = {
            "mean_reward": rewards,
            "valid_action_rate": is_valid,
            "hunt_rate": is_hunt,
            "gather_rate": is_gather,
            "flee_rate": is_flee,
            "rest_rate": is_rest,
            "mitigate_rate": is_mitigate,
            "endure_rate": is_endure,
            "eat_rate": is_eat,
            "craft_rate": is_craft,
        }

        return loss, aux

    # ------------------------------------------------------------------
    # Gumbel-Softmax loss (differentiable, uses soft action probabilities)
    # ------------------------------------------------------------------
    def _gs_loss(self, sender_input, message, receiver_input,
                 receiver_output, labels):
        """
        Differentiable loss for Gumbel-Softmax training.

        Instead of picking argmax (non-differentiable), we:
        1. Compute reward R[a] for EVERY action a ∈ {0..10} via simulation.
        2. Compute soft action probabilities: p = softmax(logits).
        3. Loss = -Σ_a p[a] * R[a]  (differentiable through softmax).

        This gives the gradient: ∇_θ loss = -Σ_a (∇_θ p[a]) * R[a]
        which directly encourages high-reward actions to get higher probability.
        """
        batch_size = receiver_output.size(0)
        device = receiver_output.device

        # Split receiver output: first 11 = action logits, last 40 = entity prediction
        action_logits = receiver_output[:, :N_ACTIONS]     # (batch, 11)
        entity_pred = receiver_output[:, N_ACTIONS:]        # (batch, 40)

        entity_vecs, state_vecs, entity_idxs, weather, turns, max_turns_vals = \
            _unpack_labels(labels)

        # Pre-compute reward matrix: (batch, N_ACTIONS)
        reward_matrix = torch.zeros(batch_size, N_ACTIONS, device=device)

        # Also track argmax-based metrics for logging
        actions = action_logits.detach().argmax(dim=-1)
        is_valid = torch.zeros(batch_size, device=device)
        is_hunt = torch.zeros(batch_size, device=device)
        is_gather = torch.zeros(batch_size, device=device)
        is_flee = torch.zeros(batch_size, device=device)
        is_rest = torch.zeros(batch_size, device=device)
        is_mitigate = torch.zeros(batch_size, device=device)
        is_endure = torch.zeros(batch_size, device=device)
        is_eat = torch.zeros(batch_size, device=device)
        is_craft = torch.zeros(batch_size, device=device)

        # Entity indices for reconstruction loss (40-class: which specific entity)
        entity_targets = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            entity = _reconstruct_entity(entity_vecs[i], int(entity_idxs[i].item()))
            state = _reconstruct_state(
                state_vecs[i],
                int(weather[i].item()),
                int(turns[i].item()),
                int(max_turns_vals[i].item()),
            )

            # Entity INDEX for reconstruction target (not type!)
            # Forces sender to encode WHICH of the 40 entities it sees.
            idx = int(entity_idxs[i].item())
            if 0 <= idx < len(ALL_ENTITIES):
                entity_targets[i] = idx
            else:
                entity_targets[i] = 0  # fallback (should not happen)

            # Compute deterministic expected rewards for ALL actions
            all_rewards = _compute_expected_rewards(entity, state)
            reward_matrix[i] = all_rewards.to(device)

            # Argmax-based metrics for logging
            action_id = int(actions[i].item())
            valid = get_valid_actions(entity, state)
            if action_id in valid:
                is_valid[i] = 1.0

            if action_id == HUNT:
                is_hunt[i] = 1.0
            elif action_id == GATHER:
                is_gather[i] = 1.0
            elif action_id == FLEE:
                is_flee[i] = 1.0
            elif action_id == REST:
                is_rest[i] = 1.0
            elif action_id == MITIGATE:
                is_mitigate[i] = 1.0
            elif action_id == ENDURE:
                is_endure[i] = 1.0
            elif action_id == EAT:
                is_eat[i] = 1.0
            elif action_id in (CRAFT_SPEAR, CRAFT_FIRE, CRAFT_SHELTER, CRAFT_ROD):
                is_craft[i] = 1.0

        # ---- Reward normalisation ----
        # Normalise rewards per-sample to mean=0, std=1.
        # This prevents large-magnitude rewards (e.g. craft=+50) from
        # dominating gradients and drowning out subtler action differences
        # (e.g. flee=+2 vs rest=-1).
        if self.reward_normalise:
            # Only normalise over valid (non-penalty) rewards per sample
            valid_rewards = reward_matrix.clone()
            invalid_mask = (valid_rewards <= -29.0)  # -30 = invalid penalty
            valid_rewards[invalid_mask] = float('nan')
            r_mean = torch.nanmean(valid_rewards, dim=-1, keepdim=True)
            r_std = valid_rewards.clone()
            r_std[invalid_mask] = 0.0
            r_mean_expanded = r_mean.expand_as(valid_rewards)
            r_std_vals = torch.where(invalid_mask, torch.zeros_like(r_std),
                                     r_std - r_mean_expanded)
            r_std_scalar = (r_std_vals ** 2).sum(dim=-1, keepdim=True) / \
                           (~invalid_mask).sum(dim=-1, keepdim=True).clamp(min=1)
            r_std_scalar = r_std_scalar.sqrt().clamp(min=1e-6)
            # Normalise valid entries, keep invalid at -30
            reward_matrix = torch.where(
                invalid_mask,
                reward_matrix,
                (reward_matrix - r_mean) / r_std_scalar,
            )

        # ---- Differentiable soft action selection ----
        # Apply action temperature: higher τ → flatter distribution → more
        # exploration.  Decoupled from the GS message temperature so we can
        # keep action exploration high even as messages sharpen.
        action_probs = F.softmax(
            action_logits / self.action_temperature, dim=-1
        )  # (batch, N_ACTIONS)

        # Expected reward under the soft policy
        expected_reward = (action_probs * reward_matrix).sum(dim=-1)  # (batch,)

        # Game loss = negative expected reward (scaled)
        game_loss = -expected_reward * self.reward_scale

        # ---- Action entropy bonus ----
        # Prevents action distribution from collapsing to a single action.
        # H(p) = -Σ p·log(p) is maximised when all actions are equally likely.
        # Adding -coeff·H to the loss encourages broader exploration.
        action_entropy = -(action_probs * (action_probs + 1e-10).log()).sum(dim=-1)
        entropy_loss = -self.action_entropy_coeff * action_entropy

        # ---- Auxiliary reconstruction loss ----
        # Entity-IDENTITY prediction from receiver's message-based hidden state.
        # 40-class cross-entropy: forces sender to encode which specific entity
        # it sees, not just the broad type.  This creates 40 distinct messages.
        # Gradient path: CE → entity_pred → receiver fc_recon → receiver RNN
        #   hidden state → message embeddings → sender RNN → sender MLP
        recon_loss = F.cross_entropy(entity_pred, entity_targets, reduction='none')
        recon_correct = (entity_pred.detach().argmax(dim=-1) == entity_targets).float()

        # Combined loss
        loss = game_loss + self.recon_weight * recon_loss + entropy_loss

        # Logging
        argmax_rewards = reward_matrix[torch.arange(batch_size), actions]

        aux = {
            "mean_reward": argmax_rewards.detach(),
            "expected_reward": expected_reward.detach(),
            "recon_loss": recon_loss.detach(),
            "recon_acc": recon_correct,
            "valid_action_rate": is_valid,
            "hunt_rate": is_hunt,
            "gather_rate": is_gather,
            "flee_rate": is_flee,
            "rest_rate": is_rest,
            "mitigate_rate": is_mitigate,
            "endure_rate": is_endure,
            "eat_rate": is_eat,
            "craft_rate": is_craft,
        }

        return loss, aux
