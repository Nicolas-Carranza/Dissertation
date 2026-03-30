#!/usr/bin/env python3
"""
Survival Game - data.py
========================

Generates training / validation / test data for the survival game.

Each sample represents ONE encounter turn:
    - sender_input:   entity vector as one-hot (6 dims x 5 values = 30 floats)
    - labels:         raw entity vector + game-state info (packed into a tensor)
    - receiver_input: game state vector (16 floats)
    - aux_input:      extra info for loss computation (entity index, weather, etc.)

The dataset pre-generates full episodes, then flattens them into individual turns.
This preserves the sequential inventory / health / energy dynamics across turns
within an episode, while fitting EGG's standard DataLoader pattern.

Data integrity:
    All samples are generated from a SINGLE pool of episodes, then split into
    train / val / test with ZERO overlap.  Episodes are assigned to splits
    BEFORE flattening, so even correlated turns within an episode stay together.
    This prevents any form of data leakage between splits.

Usage:
    from egg.zoo.survival_game.data import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(opts)
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import game mechanics from the prototype
from egg.zoo.survival_game.prototype import (
    ALL_ENTITIES,
    ANIMALS,
    CRAFT_OPPS,
    DANGERS,
    ENERGY_MAP,
    EVENTS,
    N_ACTIONS,
    RESOURCES,
    SPAWN_WEIGHTS,
    TOOL_NONE,
    VALUES_PER_DIM,
    VECTOR_DIM,
    WEATHERS,
    Entity,
    GameState,
    Inventory,
    generate_encounter,
    get_valid_actions,
    resolve_eat,
    resolve_flee,
    resolve_hunt,
    resolve_gather,
    resolve_mitigate,
    resolve_endure,
    resolve_rest,
    resolve_craft_spear,
    resolve_craft_fire,
    resolve_craft_shelter,
    resolve_craft_rod,
    update_weather,
    weather_compatible,
    ACTION_RESOLVERS,
    _apply_unaddressed_danger,
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
)


# =============================================================================
# Encoding helpers
# =============================================================================

def entity_to_onehot(entity_vector: List[int]) -> torch.Tensor:
    """
    Convert a 6-dim entity vector (each 0-4) to a one-hot representation.
    Result: 6 x 5 = 30-dimensional float tensor.
    """
    onehot = torch.zeros(VECTOR_DIM * VALUES_PER_DIM)
    for dim_idx, val in enumerate(entity_vector):
        onehot[dim_idx * VALUES_PER_DIM + val] = 1.0
    return onehot


def game_state_to_tensor(state: GameState) -> torch.Tensor:
    """
    Convert the game state to a 16-dim normalised float tensor.
    Used as the Receiver's input (what the Receiver can see without
    the Sender's message).
    """
    return torch.tensor(state.to_vector(), dtype=torch.float32)


def encode_labels(entity: Entity, state: GameState,
                  valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Pack all info needed by the loss function into a single tensor.
    Layout (26 floats):
        [0:6]   entity vector (raw ints as floats)
        [6:22]  game state vector (16 floats, normalised)
        [22]    entity index in ALL_ENTITIES (-1 if not found)
        [23]    weather (int as float)
        [24]    turn (int as float)
        [25]    max_turns (int as float)
    """
    entity_raw = torch.tensor(entity.vector, dtype=torch.float32)
    state_vec = torch.tensor(state.to_vector(), dtype=torch.float32)

    # Find entity index
    entity_idx = -1
    for i, e in enumerate(ALL_ENTITIES):
        if e.vector == entity.vector and e.name == entity.name:
            entity_idx = i
            break

    meta = torch.tensor([
        float(entity_idx),
        float(state.weather),
        float(state.turn),
        float(state.max_turns),
    ], dtype=torch.float32)

    return torch.cat([entity_raw, state_vec, meta])


# =============================================================================
# Episode simulation for data generation
# =============================================================================

def simulate_episode_for_data(
    max_turns: int = 20,
) -> List[Dict]:
    """
    Simulate one full episode using a random policy, collecting the encounter
    data at each turn.  The random policy ensures diverse game states.

    Returns a list of per-turn dicts, each containing:
        - entity: the Entity object
        - state_before: GameState snapshot (before action)
        - valid_actions: list of valid action IDs
    """
    state = GameState(max_turns=max_turns)
    turns_data = []

    for turn in range(1, max_turns + 1):
        state.turn = turn

        # Per-turn metabolic drain
        drain = random.randint(2, 4)
        state.energy -= drain

        # Weather update
        update_weather(state)

        # Generate encounter
        target = generate_encounter(state)

        # Get valid actions
        valid = get_valid_actions(target, state)

        # Snapshot the state BEFORE the action
        state_snapshot = GameState(
            energy=state.energy,
            health=state.health,
            inventory=Inventory(
                spear=state.inventory.spear,
                fire=state.inventory.fire,
                shelter=state.inventory.shelter,
                fishing_rod=state.inventory.fishing_rod,
                wood=state.inventory.wood,
                stone=state.inventory.stone,
                raw_meat=state.inventory.raw_meat,
                cooked_meat=state.inventory.cooked_meat,
                berries=state.inventory.berries,
                fish=state.inventory.fish,
                water=state.inventory.water,
                herbs=state.inventory.herbs,
            ),
            weather=state.weather,
            turn=state.turn,
            max_turns=state.max_turns,
            alive=state.alive,
        )

        turns_data.append({
            "entity": target,
            "state": state_snapshot,
            "valid_actions": valid,
        })

        # Execute a random valid action to advance the state
        action = random.choice(valid)
        resolver = ACTION_RESOLVERS.get(action)
        if resolver is not None:
            resolver(target, state)

        # Unaddressed encounter effects
        addressed = (
            (action == HUNT and target.entity_type == 0) or
            (action == GATHER and target.entity_type in (1, 3, 4)) or
            (action == MITIGATE and target.entity_type == 2) or
            (action == ENDURE and target.entity_type == 2) or
            (action == FLEE)
        )
        if not addressed:
            _apply_unaddressed_danger(target, state)

        # Clamp values
        state.energy = min(max(state.energy, 0.0), 100.0)
        state.health = min(max(state.health, 0.0), 100.0)

        # Death check
        if state.energy <= 0 or state.health <= 0:
            break

    return turns_data


# =============================================================================
# PyTorch Dataset
# =============================================================================

class SurvivalGameDataset(Dataset):
    """
    A dataset of individual encounter-turns, pre-generated from full episodes.

    Each item returns:
        (sender_input, labels, receiver_input)

    where:
        sender_input:   one-hot entity vector, 30 floats
        labels:         packed tensor with entity + state info, 26 floats
        receiver_input: game state vector (16 + 11 valid mask) = 27 floats

    Can be constructed either by generating episodes internally (legacy)
    or by accepting pre-built samples from the split-aware factory.
    """

    def __init__(
        self,
        n_episodes: int = 0,
        max_turns: int = 20,
        seed: Optional[int] = None,
        samples: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()

        if samples is not None:
            # Pre-built samples from the split-aware factory
            self.samples = samples
            return

        # Legacy path: generate episodes internally
        self.samples = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for _ in range(n_episodes):
            turns = simulate_episode_for_data(max_turns)
            for turn_data in turns:
                self.samples.append(_turn_to_sample(turn_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _turn_to_sample(
    turn_data: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a single turn dict into the (sender_input, labels, receiver_input) triple."""
    entity = turn_data["entity"]
    state = turn_data["state"]
    valid = turn_data["valid_actions"]

    sender_input = entity_to_onehot(entity.vector)
    receiver_input = game_state_to_tensor(state)

    # Valid action mask (1 = valid, 0 = invalid)
    valid_mask = torch.zeros(N_ACTIONS)
    for a in valid:
        valid_mask[a] = 1.0

    labels = encode_labels(entity, state, valid_mask)

    # receiver_input: 16 (state) + 11 (valid_mask) = 27 floats
    receiver_input_full = torch.cat([receiver_input, valid_mask])

    return (sender_input, labels, receiver_input_full)


# =============================================================================
# DataLoader factory - episode-level split (zero overlap guarantee)
# =============================================================================

def get_dataloaders(
    opts,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders with ZERO overlap guarantee.

    Strategy:
        1. Generate ALL episodes in a single pool with one seed.
        2. Shuffle episodes (not individual turns) deterministically.
        3. Split episodes into train (80%) / val (10%) / test (10%).
        4. Flatten each split's episodes into individual turn-samples.
        5. Report per-split entity coverage statistics.

    Because splits happen at the EPISODE level, even correlated turns
    within an episode always stay together - no data leakage possible.

    Expected opts attributes:
        n_episodes:        total episodes to generate (default: 10000)
        max_turns:         turns per episode (default: 20)
        batch_size:        batch size (from core.init)
        data_seed:         seed for data generation (default: 42)
        train_frac:        fraction for training (default: 0.8)
        val_frac:          fraction for validation (default: 0.1)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    n_episodes = getattr(opts, "n_episodes", 10000)
    max_turns = getattr(opts, "max_turns", 20)
    data_seed = getattr(opts, "data_seed", 42)
    train_frac = getattr(opts, "train_frac", 0.8)
    val_frac = getattr(opts, "val_frac", 0.1)
    test_frac = 1.0 - train_frac - val_frac

    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    # ---- Step 1: Generate all episodes deterministically ----
    print(f"  Generating {n_episodes} episodes (seed={data_seed})...", flush=True)
    random.seed(data_seed)
    np.random.seed(data_seed)

    all_episodes = []  # list of lists-of-turn-dicts
    for _ in range(n_episodes):
        turns = simulate_episode_for_data(max_turns)
        all_episodes.append(turns)

    # ---- Step 2: Deterministic episode shuffle ----
    rng = random.Random(data_seed)
    episode_indices = list(range(n_episodes))
    rng.shuffle(episode_indices)

    # ---- Step 3: Split at the episode level ----
    n_train = int(n_episodes * train_frac)
    n_val = int(n_episodes * val_frac)
    # n_test = remaining (avoids rounding issues)

    train_idx = episode_indices[:n_train]
    val_idx = episode_indices[n_train : n_train + n_val]
    test_idx = episode_indices[n_train + n_val :]

    # ---- Step 4: Flatten episodes into samples ----
    def _flatten(indices):
        samples = []
        for i in indices:
            for turn_data in all_episodes[i]:
                samples.append(_turn_to_sample(turn_data))
        return samples

    train_samples = _flatten(train_idx)
    val_samples = _flatten(val_idx)
    test_samples = _flatten(test_idx)

    # ---- Step 4b: Deduplicate val/test against train ----
    # Although episodes are cleanly split, the same (entity, state) pair
    # can naturally recur across independent episodes (40 entities ×
    # quantised state space → collisions by chance).  We remove any
    # val/test samples whose exact (sender_input, receiver_input) already
    # appears in the training set, for maximum scientific rigour.
    def _fingerprint(sample):
        """Create a hashable fingerprint from sender_input + receiver_input."""
        s_in, _, r_in = sample
        return (tuple(s_in.tolist()), tuple(r_in.tolist()))

    train_fps = set(_fingerprint(s) for s in train_samples)

    val_before = len(val_samples)
    val_samples = [s for s in val_samples if _fingerprint(s) not in train_fps]

    # Also build val fingerprints for test dedup
    val_fps = set(_fingerprint(s) for s in val_samples)
    combined_fps = train_fps | val_fps

    test_before = len(test_samples)
    test_samples = [s for s in test_samples if _fingerprint(s) not in combined_fps]

    val_removed = val_before - len(val_samples)
    test_removed = test_before - len(test_samples)

    if val_removed > 0 or test_removed > 0:
        print(f"  Deduplication: removed {val_removed} val samples, "
              f"{test_removed} test samples (fingerprint collisions)", flush=True)

    train_ds = SurvivalGameDataset(samples=train_samples)
    val_ds = SurvivalGameDataset(samples=val_samples)
    test_ds = SurvivalGameDataset(samples=test_samples)

    # ---- Step 5: Report statistics ----
    def _entity_coverage(ds):
        """Count unique entity vectors and entity types."""
        vecs = set()
        types = defaultdict(int)
        for s_in, labels, _ in ds.samples:
            evec = tuple(labels[:6].long().tolist())
            vecs.add(evec)
            etype = int(labels[0].item())
            types[etype] += 1
        return len(vecs), dict(types)

    type_names = {0: "Animal", 1: "Resource", 2: "Danger",
                  3: "CraftOpp", 4: "Event"}

    print(f"\n  {'Split':>8} | {'Episodes':>8} | {'Samples':>8} | "
          f"{'Entities':>8} | Entity-type distribution", flush=True)
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+{'-'*40}", flush=True)

    for name, ds, n_ep in [("Train", train_ds, len(train_idx)),
                            ("Val", val_ds, len(val_idx)),
                            ("Test", test_ds, len(test_idx))]:
        n_ent, tdict = _entity_coverage(ds)
        dist_str = "  ".join(
            f"{type_names[t]}={tdict.get(t, 0)}" for t in range(5)
        )
        print(f"  {name:>8} | {n_ep:>8} | {len(ds):>8} | "
              f"{n_ent:>5}/40  | {dist_str}", flush=True)

    print(flush=True)

    # ---- Build DataLoaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
