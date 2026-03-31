#!/usr/bin/env python3
"""
Survival Game
==============================================

This is a standalone simulator that validates the survival game mechanics BEFORE
integrating with the EGG framework. No EGG dependency - pure Python + stdlib.

Purpose:
    1. Test that the revised vector schema is consistent (all dims = 5 values)
    2. Verify game balance: random agents mostly die, smart agents mostly survive
    3. Check reward magnitudes and death-cause distributions
    4. Confirm that communication is *necessary* (Sender info > Receiver info)
    5. Validate transformation chains and inventory logic

Run:
    python3 prototype.py                          # default 500 episodes per policy
    python3 prototype.py --episodes 2000          # more episodes for tighter stats
    python3 prototype.py --episodes 5 --verbose   # see turn-by-turn details
    python3 prototype.py --seed 42                # reproducible run

Design_v2 - Revised Vector Schema:
    All 6 dimensions use exactly 5 values (0-4).
    Total combination space: 5^6 = 15,625. ~40 valid entities defined.
    See SECTION 1 below for full schema.

Authors: Nicolas Carranza Arauna
Date:    3 February 2026
"""

import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import textwrap


# ============================================================================
# SECTION 1: VECTOR SCHEMA
# ============================================================================
#
# Vector format: [entity_type, subtype, danger_level, energy_value, tool_required, weather_dependency]
#
# ALL dimensions have exactly 5 values (0–4) for consistency.
# This gives 5^6 = 15,625 possible combinations, of which ~40 are valid.
# ============================================================================

VECTOR_DIM = 6
VALUES_PER_DIM = 5

# ---------------------------------------------------------------------------
# Dimension 0 - Entity Type  (what kind of thing is it?)
# ---------------------------------------------------------------------------
ANIMAL    = 0   # Living creature, can be hunted for food
RESOURCE  = 1   # Gatherable item (food, material, water)
DANGER    = 2   # Environmental threat, must be mitigated/endured/fled
CRAFT_OPP = 3   # Crafting opportunity (good location / bonus materials)
EVENT     = 4   # Special one-off encounter (bonuses / surprises)

ENTITY_TYPE_NAMES = {
    0: "Animal", 1: "Resource", 2: "Danger",
    3: "CraftOpp", 4: "Event",
}

# ---------------------------------------------------------------------------
# Dimension 1 - Subtype  (varies by entity type, 5 per type)
# ---------------------------------------------------------------------------
SUBTYPE_NAMES = {
    ANIMAL:   {0: "predator", 1: "herbivore", 2: "reptile",
               3: "aquatic",  4: "small_game"},
    RESOURCE: {0: "raw_food", 1: "material", 2: "water",
               3: "plant",    4: "mineral"},
    DANGER:   {0: "storm",    1: "cold",     2: "hazard",
               3: "terrain",  4: "disease"},
    CRAFT_OPP:{0: "weapon",   1: "fire",     2: "shelter",
               3: "fishing",  4: "medicine"},
    EVENT:    {0: "cache",    1: "river",    2: "migration",
               3: "bounty",   4: "cave"},
}

# ---------------------------------------------------------------------------
# Dimension 2 - Danger Level  (5 levels, 0-4)
# ---------------------------------------------------------------------------
DANGER_SAFE    = 0   # 0% base injury
DANGER_LOW     = 1   # ~10% injury chance
DANGER_MEDIUM  = 2   # ~25% injury chance
DANGER_HIGH    = 3   # ~45% injury chance
DANGER_EXTREME = 4   # ~70% injury chance / death risk

DANGER_NAMES = {
    0: "safe", 1: "low", 2: "medium", 3: "high", 4: "extreme",
}

# Base injury probability per danger level (used in hunt & danger resolution)
DANGER_INJURY_PROB = {0: 0.00, 1: 0.10, 2: 0.25, 3: 0.45, 4: 0.70}

# ---------------------------------------------------------------------------
# Dimension 3 - Energy Value  (5 categories, mapped to actual numbers)
# ---------------------------------------------------------------------------
#   Category --> actual energy delta when the entity is *consumed / resolved*
ENERGY_MAP = {0: -8, 1: -3, 2: 0, 3: 3, 4: 7}

ENERGY_NAMES = {
    0: "major_drain(-8)", 1: "minor_drain(-3)", 2: "neutral(0)",
    3: "low_gain(+3)",    4: "high_gain(+7)",
}

# ---------------------------------------------------------------------------
# Dimension 4 - Tool Required  (which tool is needed to interact optimally)
# ---------------------------------------------------------------------------
TOOL_NONE    = 0   # Bare hands / no tool needed
TOOL_SPEAR   = 1   # Hunting weapon (wood + stone)
TOOL_ROD     = 2   # Fishing rod   (wood)
TOOL_FIRE    = 3   # Fire          (wood)
TOOL_SHELTER = 4   # Shelter       (wood × 2)

TOOL_NAMES = {
    0: "none", 1: "spear", 2: "fishing_rod", 3: "fire", 4: "shelter",
}

# ---------------------------------------------------------------------------
# Dimension 5 - Weather Dependency  (preferred / required weather)
# ---------------------------------------------------------------------------
WEATHER_ANY   = 0   # Works in all weather
WEATHER_SUNNY = 1   # Better/only in sunny + calm
WEATHER_RAIN  = 2   # Better/only in rain
WEATHER_CALM  = 3   # Needs calm weather specifically
WEATHER_NIGHT = 4   # Night-time encounter

WEATHER_NAMES = {
    0: "any", 1: "sunny", 2: "rain", 3: "calm", 4: "night",
}

# Current-weather states  (the world cycles through these)
WEATHERS = [0, 1, 2, 3, 4]  # calm, sunny, rain, storm, night
WEATHER_STATE_NAMES = {
    0: "calm", 1: "sunny", 2: "rain", 3: "stormy", 4: "night",
}


# ============================================================================
# SECTION 2: ENTITY CATALOG
# ============================================================================
# Each entity: (name, vector, description, inventory_key, inventory_qty)
#   inventory_key / qty = what you get when you GATHER this entity.
#   For animals, inventory_key = "raw_meat" (after successful hunt).
#   For dangers/events, inventory_key is the bonus you receive.
# ============================================================================

@dataclass
class Entity:
    """One thing the agents can encounter in the world."""
    name: str
    vector: List[int]       # [entity_type, subtype, danger, energy, tool_req, weather]
    description: str = ""
    inventory_key: str = "" # What inventory slot this yields (gather/hunt)
    inventory_qty: int = 1  # How many units

    # ---- convenience accessors ----
    @property
    def entity_type(self):   return self.vector[0]
    @property
    def subtype(self):       return self.vector[1]
    @property
    def danger_level(self):  return self.vector[2]
    @property
    def energy_category(self): return self.vector[3]
    @property
    def energy_value(self):  return ENERGY_MAP[self.vector[3]]
    @property
    def tool_required(self): return self.vector[4]
    @property
    def weather_dep(self):   return self.vector[5]

    def short(self):
        return (f"{self.name} {self.vector} "
                f"[{ENTITY_TYPE_NAMES[self.entity_type]}/"
                f"{SUBTYPE_NAMES[self.entity_type][self.subtype]}]")


# ---- Animals  [0, sub, danger, energy, tool, weather] ---------------------
ANIMALS = [
    Entity("Lion",     [0,0,4,4,1,0], "Predator. Extreme danger, high reward, spear.",
           "raw_meat", 2),
    Entity("Wolf",     [0,0,3,3,1,0], "Pack predator. High danger, medium reward.",
           "raw_meat", 1),
    Entity("Bear",     [0,0,4,4,1,3], "Large predator. Extreme danger, calm weather.",
           "raw_meat", 2),
    Entity("Deer",     [0,1,1,4,1,1], "Herbivore. Low danger, high reward, sunny.",
           "raw_meat", 2),
    Entity("Goat",     [0,1,1,3,1,0], "Herbivore. Low danger, medium reward.",
           "raw_meat", 1),
    Entity("Snake",    [0,2,3,2,0,0], "Reptile. High danger, low reward, bare hands.",
           "raw_meat", 1),
    Entity("Lizard",   [0,2,1,3,0,1], "Reptile. Low danger, medium reward, sunny.",
           "raw_meat", 1),
    Entity("Fish",     [0,3,0,3,2,2], "Aquatic. Safe, medium reward, rod, rain.",
           "fish", 1),
    Entity("Salmon",   [0,3,0,4,2,2], "Aquatic. Safe, high reward, rod, rain.",
           "fish", 2),
    Entity("Rabbit",   [0,4,1,3,1,1], "Small game. Low danger, medium, spear, sunny.",
           "raw_meat", 1),
    Entity("Squirrel", [0,4,0,2,0,1], "Small game. Safe, low reward, bare hands, sunny.",
           "raw_meat", 1),
]

# ---- Resources  [1, sub, danger, energy, tool, weather] -------------------
RESOURCES = [
    Entity("Berries",    [1,3,1,3,0,1], "Plant. Low poison risk, medium energy, sunny.",
           "berries", 2),
    Entity("Mushrooms",  [1,3,2,3,0,0], "Plant. Medium poison risk, medium energy.",
           "berries", 1),  # treated same as berries in inventory
    Entity("Herbs",      [1,3,0,2,0,1], "Plant. Safe, neutral energy, sunny.",
           "herbs", 2),
    Entity("Wood",       [1,1,1,2,0,1], "Material. Low danger, sunny preferred.",
           "wood", 2),
    Entity("Firewood",   [1,1,0,2,0,0], "Material. Safe, any weather.",
           "wood", 1),
    Entity("Stone",      [1,4,0,2,0,0], "Mineral. Safe, any weather.",
           "stone", 2),
    Entity("Flint",      [1,4,1,2,0,0], "Mineral. Low danger, sharp edges.",
           "stone", 1),
    Entity("Water",      [1,2,0,3,0,0], "Water. Safe, medium energy.",
           "water", 2),
    Entity("Muddy Water",[1,2,2,3,0,2], "Water. Medium disease risk, rain.",
           "water", 1),
]

# ---- Dangers  [2, sub, danger, energy, tool_needed, weather] ---------------
DANGERS = [
    Entity("Storm",        [2,0,3,0,4,2], "Weather. High, needs shelter, rain."),
    Entity("Blizzard",     [2,0,4,0,4,0], "Weather. Extreme, needs shelter."),
    Entity("Cold Night",   [2,1,2,1,3,4], "Cold. Medium, needs fire, night."),
    Entity("Frost",        [2,1,3,0,3,4], "Cold. High, needs fire, night."),
    Entity("Poison Plant", [2,2,3,0,0,0], "Hazard. High danger, no tool helps."),
    Entity("Thorns",       [2,2,2,1,0,1], "Hazard. Medium, sunny."),
    Entity("Cliff",        [2,3,3,1,0,0], "Terrain. High danger."),
    Entity("Quicksand",    [2,3,4,0,0,2], "Terrain. Extreme, rain."),
    Entity("Fever",        [2,4,2,1,0,0], "Disease. Medium danger."),
    Entity("Infection",    [2,4,3,0,0,0], "Disease. High danger."),
]

# ---- Crafting Opportunities  [3, sub, danger, energy, tool, weather] -------
#   These represent finding a good spot / bonus materials for crafting.
#   GATHER here gives bonus materials; crafting is more efficient.
CRAFT_OPPS = [
    Entity("Weapon Cache", [3,0,0,2,0,0], "Good spot with flint & sticks.",
           "stone", 1),
    Entity("Dry Campsite", [3,1,0,2,0,1], "Ideal fire spot, sunny.",
           "wood", 1),
    Entity("Rocky Outcrop",[3,2,0,2,0,3], "Natural shelter base, calm.",
           "wood", 2),
    Entity("Riverbank",    [3,3,0,2,0,2], "Good fishing spot, rain.",
           "wood", 1),
    Entity("Herb Garden",  [3,4,0,2,0,1], "Medicinal herbs, sunny.",
           "herbs", 2),
]

# ---- Events  [4, sub, danger, energy, tool, weather] ----------------------
#   One-off encounters with generally positive outcomes.
EVENTS = [
    Entity("Supply Cache",  [4,0,0,3,0,0], "Abandoned supplies!",
           "wood", 2),
    Entity("River Crossing",[4,1,1,2,0,0], "Ford a river. Low risk.",
           "water", 2),
    Entity("Animal Migration",[4,2,0,3,1,0],"Herd passing. Easy hunting.",
           "raw_meat", 1),
    Entity("Berry Patch",   [4,3,0,4,0,1], "Large bush. High energy, sunny.",
           "berries", 3),
    Entity("Cave",          [4,4,0,2,0,4], "Natural cave. Safe shelter, night.",
           "wood", 1),  # find dry wood inside
]

ALL_ENTITIES = ANIMALS + RESOURCES + DANGERS + CRAFT_OPPS + EVENTS

# Spawn weights: probability of each entity TYPE appearing per turn.
# These are relative weights; they don't need to sum to 100.
SPAWN_WEIGHTS = {
    ANIMAL:   25,
    RESOURCE: 35,
    DANGER:   20,
    CRAFT_OPP:10,
    EVENT:    10,
}


# ============================================================================
# SECTION 3: INVENTORY & GAME STATE
# ============================================================================

@dataclass
class Inventory:
    """
    Joint inventory shared by both agents.
    Each item type has a maximum capacity.  Overflow is lost.
    """
    # ---- Tools (boolean: have it or not, persistent once crafted) ----
    spear:       bool = False
    fire:        bool = False
    shelter:     bool = False
    fishing_rod: bool = False

    # ---- Materials (integer counts, capped) ----
    wood:  int = 0    # cap 5
    stone: int = 0    # cap 3

    # ---- Food & consumables (integer counts, capped) ----
    raw_meat:    int = 0   # cap 3
    cooked_meat: int = 0   # cap 3
    berries:     int = 0   # cap 3
    fish:        int = 0   # cap 3
    water:       int = 0   # cap 3
    herbs:       int = 0   # cap 3

    # ---- Capacity constants ----
    MAX_WOOD  = 5
    MAX_STONE = 3
    MAX_FOOD  = 3   # per food type

    def cap(self, key: str) -> int:
        """Return the max capacity for a given inventory key."""
        if key == "wood":  return self.MAX_WOOD
        if key == "stone": return self.MAX_STONE
        return self.MAX_FOOD  # all food / consumable types

    def get(self, key: str) -> int:
        """Get current count of an inventory item."""
        return getattr(self, key, 0)

    def add(self, key: str, qty: int) -> int:
        """
        Add qty units.  Returns how many were actually added (rest overflow).
        """
        current = self.get(key)
        space = self.cap(key) - current
        added = min(qty, max(space, 0))
        setattr(self, key, current + added)
        return added

    def remove(self, key: str, qty: int) -> int:
        """Remove qty units.  Returns how many were actually removed."""
        current = self.get(key)
        removed = min(qty, current)
        setattr(self, key, current - removed)
        return removed

    def has_tool(self, tool_id: int) -> bool:
        """Check whether we possess the tool identified by its vector-dim-4 ID."""
        return {
            TOOL_NONE:    True,
            TOOL_SPEAR:   self.spear,
            TOOL_ROD:     self.fishing_rod,
            TOOL_FIRE:    self.fire,
            TOOL_SHELTER: self.shelter,
        }.get(tool_id, False)

    def total_food(self) -> int:
        return (self.raw_meat + self.cooked_meat + self.berries
                + self.fish + self.water + self.herbs)

    def best_food(self) -> Optional[str]:
        """Return the key of the best food to eat (highest energy first)."""
        # Priority: cooked_meat(+15) > fish(+10) > berries(+6) > water(+5)
        #           > raw_meat(+5 but parasite risk) > herbs(+3)
        priority = [
            ("cooked_meat", 15), ("fish", 10), ("berries", 6),
            ("water", 5), ("raw_meat", 5), ("herbs", 3),
        ]
        for key, _ in priority:
            if self.get(key) > 0:
                return key
        return None

    def to_vector(self) -> List[float]:
        """Normalised vector for agent input (16 floats)."""
        return [
            float(self.spear), float(self.fire),
            float(self.shelter), float(self.fishing_rod),
            self.wood  / self.MAX_WOOD,
            self.stone / self.MAX_STONE,
            self.raw_meat    / self.MAX_FOOD,
            self.cooked_meat / self.MAX_FOOD,
            self.berries     / self.MAX_FOOD,
            self.fish        / self.MAX_FOOD,
            self.water       / self.MAX_FOOD,
            self.herbs       / self.MAX_FOOD,
        ]

    def summary(self) -> str:
        tools = []
        if self.spear:       tools.append("spear")
        if self.fire:        tools.append("fire")
        if self.shelter:     tools.append("shelter")
        if self.fishing_rod: tools.append("rod")
        t = ",".join(tools) if tools else "none"
        return (f"tools=[{t}] wood={self.wood} stone={self.stone} "
                f"raw_meat={self.raw_meat} cooked={self.cooked_meat} "
                f"berries={self.berries} fish={self.fish} "
                f"water={self.water} herbs={self.herbs}")


@dataclass
class GameState:
    """
    Full mutable state for one episode.
    Both agents share this state (cooperative game).
    """
    energy:    float = 100.0   # 0–100  (0 = starvation death)
    health:    float = 100.0   # 0–100  (0 = injury death)
    inventory: Inventory = field(default_factory=Inventory)
    weather:   int   = 1       # current weather state (0-4)
    turn:      int   = 0
    max_turns: int   = 20
    alive:     bool  = True
    total_reward: float = 0.0
    death_cause:  str   = ""

    # ---- Tracking / analytics ----
    actions_taken:   Dict = field(default_factory=lambda: defaultdict(int))
    tools_crafted:   List = field(default_factory=list)
    hunts_attempted: int  = 0
    hunts_successful:int  = 0
    chains_completed:int  = 0
    food_eaten:      int  = 0
    dangers_faced:   int  = 0
    dangers_mitigated:int = 0

    def to_vector(self) -> List[float]:
        """
        State vector visible to both agents: 16 floats.
        [energy, health, weather, turn_progress, ...12 inventory dims...]
        """
        return [
            self.energy  / 100.0,
            self.health  / 100.0,
            self.weather / 4.0,
            self.turn    / max(self.max_turns, 1),
        ] + self.inventory.to_vector()

    def status_line(self) -> str:
        return (f"T{self.turn:>2}/{self.max_turns} "
                f"E={self.energy:5.1f} H={self.health:5.1f} "
                f"W={WEATHER_STATE_NAMES[self.weather]:>6} "
                f"R={self.total_reward:+.0f} | {self.inventory.summary()}")


# ============================================================================
# SECTION 4: ACTION SPACE
# ============================================================================
#
# Fixed set of 11 actions.  The Receiver always picks from this set.
# Invalid actions (wrong encounter type, missing materials) yield a
# "wasted turn" penalty.  Unaddressed dangers still deal damage.
#
# ============================================================================

HUNT          = 0    # Attack an animal
GATHER        = 1    # Collect a resource / event / craft-opp bonus
FLEE          = 2    # Run away (any encounter; safe but costs energy)
REST          = 3    # Skip turn (small energy recovery)
MITIGATE      = 4    # Use matching tool against a danger
ENDURE        = 5    # Brace through danger (reduced damage)
EAT           = 6    # Consume best food from inventory
CRAFT_SPEAR   = 7    # wood ≥1 + stone ≥1  →  spear
CRAFT_FIRE    = 8    # wood ≥1              →  fire
CRAFT_SHELTER = 9    # wood ≥2              →  shelter
CRAFT_ROD     = 10   # wood ≥1              →  fishing_rod

N_ACTIONS = 11

ACTION_NAMES = {
    0: "hunt",     1: "gather",       2: "flee",
    3: "rest",     4: "mitigate",     5: "endure",
    6: "eat",      7: "craft_spear",  8: "craft_fire",
    9: "craft_shelter", 10: "craft_rod",
}


def get_valid_actions(entity: Entity, state: GameState) -> List[int]:
    """
    Return the list of *valid* action IDs given the current encounter and
    game state.  An action is valid if its preconditions are met.
    """
    valid = [FLEE, REST]          # always available
    inv = state.inventory

    # ---- Encounter-dependent ----
    if entity.entity_type == ANIMAL:
        valid.append(HUNT)
    elif entity.entity_type == RESOURCE:
        valid.append(GATHER)
    elif entity.entity_type == DANGER:
        valid.append(ENDURE)
        # tool_required=0 means no tool can help; only endure/flee
        if entity.tool_required != TOOL_NONE and inv.has_tool(entity.tool_required):
            valid.append(MITIGATE)
    elif entity.entity_type in (CRAFT_OPP, EVENT):
        valid.append(GATHER)

    # ---- Inventory-dependent (always checkable) ----
    if inv.total_food() > 0:
        valid.append(EAT)
    if inv.wood >= 1 and inv.stone >= 1 and not inv.spear:
        valid.append(CRAFT_SPEAR)
    if inv.wood >= 1 and not inv.fire:
        valid.append(CRAFT_FIRE)
    if inv.wood >= 2 and not inv.shelter:
        valid.append(CRAFT_SHELTER)
    if inv.wood >= 1 and not inv.fishing_rod:
        valid.append(CRAFT_ROD)

    return sorted(set(valid))


# ============================================================================
# SECTION 5: ACTION RESOLUTION
# ============================================================================
#
# Each resolver returns (reward_delta, description_string).
# Side-effects are applied directly to `state`.
#
# IMPORTANT RULE - Unaddressed encounters:
#   • Animal (danger ≥ 3) that you don't HUNT or FLEE → it attacks you.
#   • Danger that you don't MITIGATE / ENDURE / FLEE → full damage.
#   • Resource / CraftOpp / Event you ignore → opportunity lost, no penalty.
# ============================================================================

# ---- Danger damage tables (by danger_level) ----
UNMITIGATED_HEALTH = {0:  0, 1:  5, 2: 12, 3: 20, 4: 30}
UNMITIGATED_ENERGY = {0:  0, 1:  3, 2:  6, 3: 10, 4: 15}
ENDURE_FACTOR      = 0.6   # endure reduces damage to 60%
MITIGATE_FLAT      = 2     # mitigate with correct tool: only 2 hp / 2 energy


def _apply_unaddressed_danger(entity: Entity, state: GameState) -> float:
    """Apply full danger damage when the agent didn't address it.  Returns penalty."""
    if entity.entity_type != DANGER:
        # Aggressive animals attack if ignored
        if entity.entity_type == ANIMAL and entity.danger_level >= 3:
            dmg = 5 + entity.danger_level * 7
            state.health -= dmg
            return -dmg * 0.5    # penalty proportional to damage
        return 0.0

    state.dangers_faced += 1
    h_loss = UNMITIGATED_HEALTH[entity.danger_level]
    e_loss = UNMITIGATED_ENERGY[entity.danger_level]
    state.health -= h_loss
    state.energy -= e_loss
    return -(h_loss + e_loss) * 0.5


def resolve_hunt(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Attempt to hunt an animal."""
    if entity.entity_type != ANIMAL:
        return -10.0, "WASTED: tried to hunt a non-animal"

    state.hunts_attempted += 1
    inv = state.inventory
    has_tool = inv.has_tool(entity.tool_required)

    # ---- success rate ----
    if has_tool and entity.danger_level <= 1:
        success = 0.85
    elif has_tool:
        success = 0.65
    elif entity.tool_required == TOOL_NONE:
        success = 0.50
    else:
        success = 0.15

    success *= (state.health / 100.0) * (min(state.energy, 50) / 50.0)

    reward = 0.0
    if random.random() < success:
        # ---- hunt success ----
        state.hunts_successful += 1
        added = inv.add(entity.inventory_key, entity.inventory_qty)
        reward = 15 + entity.danger_level * 10
        if added < entity.inventory_qty:
            reward -= 5   # partial - inventory overflow
        # small injury risk even on success
        if random.random() < entity.danger_level * 0.10:
            dmg = 5 + entity.danger_level * 5
            state.health -= dmg
            reward -= 5
        state.energy -= 8
        return reward, f"HUNT SUCCESS: +{added} {entity.inventory_key}"
    else:
        # ---- hunt failure ----
        reward = -5
        if random.random() < entity.danger_level * 0.25:
            dmg = 10 + entity.danger_level * 8
            state.health -= dmg
            reward -= 15
        state.energy -= 12
        return reward, "HUNT FAIL"


def resolve_gather(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Collect a resource, event loot, or crafting-opportunity bonus."""
    if entity.entity_type not in (RESOURCE, CRAFT_OPP, EVENT):
        return -10.0, "WASTED: tried to gather from animal/danger"

    key = entity.inventory_key
    qty = entity.inventory_qty
    if not key:
        return 0.0, "GATHER: nothing to pick up"

    added = state.inventory.add(key, qty)
    state.energy -= 2   # small gathering cost

    # Poison/parasite risk for certain resources
    poison_penalty = 0.0
    if entity.danger_level >= 2 and random.random() < DANGER_INJURY_PROB[entity.danger_level]:
        dmg = 5 + entity.danger_level * 5
        state.health -= dmg
        poison_penalty = -dmg * 0.5

    reward = 5.0 + added * 2.0 + poison_penalty
    overflow = qty - added
    desc = f"GATHER: +{added} {key}"
    if overflow > 0:
        desc += f" (lost {overflow}, inv full)"
        reward -= 3
    return reward, desc


def resolve_flee(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Run away from the encounter.  Always safe, costs energy."""
    state.energy -= 5
    # Slight positive reward for fleeing extreme danger, penalty otherwise
    if entity.entity_type == DANGER and entity.danger_level >= 3:
        return 2.0, "FLEE (wise)"
    if entity.entity_type == ANIMAL and entity.danger_level >= 3:
        return 1.0, "FLEE (safe)"
    return -2.0, "FLEE (opportunity lost)"


def resolve_rest(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Do nothing with the encounter.  Recover a bit of energy."""
    state.energy += 3
    return -1.0, "REST"


def resolve_mitigate(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Use the correct tool to mitigate a danger."""
    if entity.entity_type != DANGER:
        return -10.0, "WASTED: mitigate on non-danger"
    state.dangers_faced += 1
    if not state.inventory.has_tool(entity.tool_required):
        # Don't have the right tool - same as endure
        return resolve_endure(entity, state)

    state.dangers_mitigated += 1
    state.health -= MITIGATE_FLAT
    state.energy -= MITIGATE_FLAT

    # Bonus for fire-vs-cold
    bonus = 0.0
    if entity.tool_required == TOOL_FIRE and entity.subtype == 1:  # cold
        state.energy += 5
        bonus = 5.0

    reward = 20.0 + entity.danger_level * 5 + bonus
    state.chains_completed += 1   # mitigating with a crafted tool = chain
    return reward, f"MITIGATE with {TOOL_NAMES[entity.tool_required]} (+bonus={bonus:.0f})"


def resolve_endure(entity: Entity, state: GameState) -> Tuple[float, str]:
    """Brace through a danger, taking reduced damage."""
    if entity.entity_type != DANGER:
        return -10.0, "WASTED: endure on non-danger"
    state.dangers_faced += 1
    h_loss = int(UNMITIGATED_HEALTH[entity.danger_level] * ENDURE_FACTOR)
    e_loss = int(UNMITIGATED_ENERGY[entity.danger_level] * ENDURE_FACTOR)
    state.health -= h_loss
    state.energy -= e_loss
    reward = -(h_loss + e_loss) * 0.3
    return reward, f"ENDURE: -{h_loss}hp -{e_loss}en"


def resolve_eat(entity: Entity, state: GameState) -> Tuple[float, str]:
    """
    Consume the best available food from inventory.
    If we have fire + raw_meat, auto-cook first (transformation chain!).
    """
    inv = state.inventory

    # Auto-cook: raw_meat + fire → cooked_meat
    if inv.fire and inv.raw_meat > 0 and inv.cooked_meat < Inventory.MAX_FOOD:
        inv.raw_meat -= 1
        inv.cooked_meat += 1
        state.chains_completed += 1   # transformation chain completed

    food_key = inv.best_food()
    if food_key is None:
        return -5.0, "EAT: no food!"

    inv.remove(food_key, 1)
    state.food_eaten += 1

    # Energy gained depends on food type
    # Values are high so eating covers multiple turns of metabolic drain.
    # This makes the gather→craft→hunt→cook→eat chain strategically valuable.
    food_energy = {
        "cooked_meat": 15, "fish": 10, "berries": 6,
        "water": 5, "raw_meat": 5, "herbs": 3,
    }
    gained = food_energy.get(food_key, 1)
    state.energy += gained

    # Parasite risk from raw meat
    penalty = 0.0
    if food_key == "raw_meat" and random.random() < 0.30:
        state.health -= 12
        penalty = -12.0

    reward = gained + penalty
    # Bonus for eating cooked meat (completed chain)
    if food_key == "cooked_meat":
        reward += 5.0

    return reward, f"EAT {food_key}: +{gained}en" + (
        " (PARASITES!)" if penalty < 0 else "")


def _resolve_craft(tool_name: str, inv: Inventory, state: GameState,
                    wood_cost: int, stone_cost: int, tool_attr: str
                    ) -> Tuple[float, str]:
    """Generic craft resolution."""
    if getattr(inv, tool_attr):
        return -5.0, f"WASTED: already have {tool_name}"
    if inv.wood < wood_cost or inv.stone < stone_cost:
        return -10.0, f"WASTED: not enough materials for {tool_name}"

    inv.wood  -= wood_cost
    inv.stone -= stone_cost
    setattr(inv, tool_attr, True)
    state.tools_crafted.append(tool_name)
    state.energy -= 3
    return 50.0, f"CRAFT {tool_name}! (discovery bonus)"


def resolve_craft_spear(entity, state):
    return _resolve_craft("spear", state.inventory, state, 1, 1, "spear")

def resolve_craft_fire(entity, state):
    return _resolve_craft("fire", state.inventory, state, 1, 0, "fire")

def resolve_craft_shelter(entity, state):
    return _resolve_craft("shelter", state.inventory, state, 2, 0, "shelter")

def resolve_craft_rod(entity, state):
    return _resolve_craft("fishing_rod", state.inventory, state, 1, 0, "fishing_rod")


# ---- Dispatch table ----
ACTION_RESOLVERS = {
    HUNT:          resolve_hunt,
    GATHER:        resolve_gather,
    FLEE:          resolve_flee,
    REST:          resolve_rest,
    MITIGATE:      resolve_mitigate,
    ENDURE:        resolve_endure,
    EAT:           resolve_eat,
    CRAFT_SPEAR:   resolve_craft_spear,
    CRAFT_FIRE:    resolve_craft_fire,
    CRAFT_SHELTER: resolve_craft_shelter,
    CRAFT_ROD:     resolve_craft_rod,
}


def compute_step_reward(
    action_id: int,
    target: Entity,
    state: GameState,
    valid_actions: Optional[List[int]] = None,
) -> Tuple[float, str, bool]:
    """
    Compute one-step reward using the same mechanics used by agent evaluation.

    Returns:
        reward: scalar turn reward after bonuses/penalties
        desc: action outcome description
        was_valid: whether action was valid for the encounter/state
    """
    valid = valid_actions if valid_actions is not None else get_valid_actions(target, state)
    was_valid = action_id in valid

    resolver = ACTION_RESOLVERS.get(action_id)
    if resolver is None or not was_valid:
        reward = -15.0
        desc = "INVALID"
    else:
        reward, desc = resolver(target, state)

    addressed = (
        (action_id == HUNT and target.entity_type == ANIMAL) or
        (action_id == GATHER and target.entity_type in (RESOURCE, CRAFT_OPP, EVENT)) or
        (action_id == MITIGATE and target.entity_type == DANGER) or
        (action_id == ENDURE and target.entity_type == DANGER) or
        (action_id == FLEE)
    )
    if not addressed:
        reward += _apply_unaddressed_danger(target, state)

    # Per-turn survival shaping (kept consistent with callbacks evaluator path)
    reward += 10.0
    if state.energy > 70:
        reward += 5.0
    if state.health > 70:
        reward += 3.0

    return reward, desc, was_valid


# ============================================================================
# SECTION 6: ENCOUNTER GENERATION
# ============================================================================

def weather_compatible(entity: Entity, current_weather: int) -> bool:
    """Check if an entity can appear in the current weather."""
    dep = entity.weather_dep
    if dep == WEATHER_ANY:
        return True
    if dep == WEATHER_SUNNY and current_weather in (0, 1):   # calm or sunny
        return True
    if dep == WEATHER_RAIN  and current_weather in (2, 3):   # rain or stormy
        return True
    if dep == WEATHER_CALM  and current_weather == 0:        # calm only
        return True
    if dep == WEATHER_NIGHT and current_weather == 4:        # night only
        return True
    return False


def generate_encounter(state: GameState) -> Entity:
    """
    Generate one target entity for this turn.

    1. Pick entity TYPE by spawn weights.
    2. Filter entities of that type by weather compatibility.
    3. Pick one target at random.
    """
    # Build weather-filtered pools per entity type
    pools = defaultdict(list)
    for e in ALL_ENTITIES:
        if weather_compatible(e, state.weather):
            pools[e.entity_type].append(e)

    # Weighted type selection (only from types with available entities)
    available_types = [t for t in SPAWN_WEIGHTS if pools[t]]
    if not available_types:
        # Fallback: something always spawnable
        available_types = [RESOURCE]
        pools[RESOURCE] = [e for e in RESOURCES if e.weather_dep == WEATHER_ANY]

    weights = [SPAWN_WEIGHTS[t] for t in available_types]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0
    chosen_type = available_types[0]
    for t, w in zip(available_types, weights):
        cumulative += w
        if r <= cumulative:
            chosen_type = t
            break

    return random.choice(pools[chosen_type])


def update_weather(state: GameState):
    """Cycle weather every 4-6 turns."""
    if state.turn > 0 and state.turn % random.randint(4, 6) == 0:
        state.weather = random.choice(WEATHERS)


# ============================================================================
# SECTION 7: EPISODE RUNNER
# ============================================================================

def run_episode(policy_fn, max_turns: int = 20,
                verbose: bool = False, add_completion_bonus: bool = False) -> GameState:
    """
    Simulate one full episode using the given policy function.

    Args:
        policy_fn:  callable(entity, state, valid_actions) → action_id
        max_turns:  episode length
        verbose:    print turn-by-turn log

    Returns:
        Final GameState with all tracking info.
    """
    state = GameState(max_turns=max_turns)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  NEW EPISODE  (max_turns={max_turns})")
        print(f"{'='*70}")

    for turn in range(1, max_turns + 1):
        state.turn = turn

        # ---- 1. Per-turn energy drain (metabolism) ----
        drain = random.randint(2, 4)
        state.energy -= drain

        # ---- 2. Weather update ----
        update_weather(state)

        # ---- 3. Generate encounter ----
        target = generate_encounter(state)

        # ---- 4. Get valid actions ----
        valid = get_valid_actions(target, state)

        # ---- 5. Agent picks action ----
        action = policy_fn(target, state, valid)
        state.actions_taken[action] += 1

        # ---- 6. Resolve action + per-turn reward ----
        turn_reward, desc, _was_valid = compute_step_reward(
            action, target, state, valid_actions=valid
        )
        state.total_reward += turn_reward

        if verbose:
            print(f"  {state.status_line()}")
            print(f"    Encounter: {target.short()}")
            print(f"    Valid: {[ACTION_NAMES[a] for a in valid]}")
            print(f"    Action:  {ACTION_NAMES.get(action, '???')} → {desc}")
            print(f"    Turn reward: {turn_reward:+.1f}")

        # ---- 9. Death check ----
        state.energy = min(max(state.energy, 0.0), 100.0)
        state.health = min(max(state.health, 0.0), 100.0)

        if state.energy <= 0:
            state.alive = False
            state.death_cause = "starvation"
            state.total_reward -= 50
            if verbose:
                print(f"    ☠ DIED: starvation (energy={state.energy:.1f})")
            break
        if state.health <= 0:
            state.alive = False
            state.death_cause = "injury"
            state.total_reward -= 300
            if verbose:
                print(f"    ☠ DIED: injury (health={state.health:.1f})")
            break

    # ---- Optional legacy completion bonus (disabled by default for comparability) ----
    if state.alive and add_completion_bonus:
        completion_bonus = 150.0
        efficiency = (state.energy + state.health) / 2.0
        state.total_reward += completion_bonus + efficiency
        if verbose:
            print(f"\n  ✓ SURVIVED!  completion=+150  efficiency=+{efficiency:.0f}")
            print(f"  Final reward: {state.total_reward:+.1f}")

    if verbose:
        print(f"  Tools crafted: {state.tools_crafted}")
        print(f"  Hunts: {state.hunts_successful}/{state.hunts_attempted}")
        print(f"  Chains completed: {state.chains_completed}")
        print(f"  Food eaten: {state.food_eaten}")
        print(f"  Dangers faced/mitigated: {state.dangers_faced}/{state.dangers_mitigated}")

    return state


# ============================================================================
# SECTION 8: AGENT POLICIES
# ============================================================================
#
# Three policies to test game balance:
#   1. RANDOM  - picks uniformly from valid actions (lower bound)
#   2. GREEDY  - heuristic: address encounter if possible, else best inv action
#   3. OPTIMAL - stronger heuristic: considers future state
# ============================================================================

def random_policy(entity, state, valid_actions):
    """Pick a random valid action."""
    return random.choice(valid_actions)


def greedy_policy(entity, state, valid_actions):
    """
    Simple priority-based heuristic:
      1. If danger → mitigate (if possible) else endure
      2. If energy < 25 and have food → eat
      3. If animal is safe-ish + have spear → hunt
      4. If resource → gather (if inventory has space)
      5. If can craft something useful → craft (priority: fire > spear > shelter > rod)
      6. If animal without spear → flee
      7. Else rest
    """
    inv = state.inventory
    etype = entity.entity_type

    # Priority 1: address dangers
    if etype == DANGER:
        if MITIGATE in valid_actions:
            return MITIGATE
        if FLEE in valid_actions and entity.danger_level >= 3:
            return FLEE
        if ENDURE in valid_actions:
            return ENDURE
        return FLEE

    # Priority 2: critical energy → eat
    if state.energy < 25 and EAT in valid_actions:
        return EAT

    # Priority 3: hunt if reasonable
    if etype == ANIMAL and HUNT in valid_actions:
        # Only hunt if we have the tool or danger is low
        if inv.has_tool(entity.tool_required) or entity.danger_level <= 1:
            return HUNT
        elif entity.danger_level >= 3:
            return FLEE
        # medium danger without tool - risky, skip
        pass

    # Priority 4: craft essential tools BEFORE gathering more
    if CRAFT_FIRE in valid_actions and not inv.fire:
        return CRAFT_FIRE
    if CRAFT_SPEAR in valid_actions and not inv.spear:
        return CRAFT_SPEAR

    # Priority 5: gather resources
    if etype in (RESOURCE, CRAFT_OPP, EVENT) and GATHER in valid_actions:
        return GATHER

    # Priority 6: craft remaining tools
    if CRAFT_SHELTER in valid_actions and not inv.shelter:
        return CRAFT_SHELTER
    if CRAFT_ROD in valid_actions and not inv.fishing_rod:
        return CRAFT_ROD

    # Priority 6: eat if moderately low
    if state.energy < 50 and EAT in valid_actions:
        return EAT

    # Priority 7: flee from aggressive animals
    if etype == ANIMAL and entity.danger_level >= 2:
        return FLEE

    # Default
    return REST


def optimal_policy(entity, state, valid_actions):
    """
    Best heuristic policy - models informed agent with strategic planning.

    Key improvements over GREEDY:
      - Invests in tool crafting earlier (fire first for cooking chain)
      - Hunts more aggressively when equipped
      - Eats proactively at higher thresholds to avoid starvation spirals
      - Flees only extreme threats, endures moderate ones
      - Prioritises high-value gathers (wood/stone for tools, food for survival)
    """
    inv = state.inventory
    etype = entity.entity_type

    # ---- PANIC: about to die ----
    if state.energy < 15 and EAT in valid_actions:
        return EAT
    if state.health < 15 and etype in (DANGER, ANIMAL) and entity.danger_level >= 2:
        return FLEE

    # ---- DANGER: always address immediately ----
    if etype == DANGER:
        if MITIGATE in valid_actions:
            return MITIGATE
        if entity.danger_level >= 4:
            return FLEE   # extreme = flee rather than endure
        if ENDURE in valid_actions:
            return ENDURE
        return FLEE

    # ---- CRAFT essential tools ASAP (fire > spear) ----
    #   Fire enables cooking chain (+15 energy per cooked meal).
    #   Spear enables efficient hunting.
    if CRAFT_FIRE in valid_actions and not inv.fire:
        return CRAFT_FIRE
    if CRAFT_SPEAR in valid_actions and not inv.spear:
        return CRAFT_SPEAR

    # ---- EAT if energy getting low ----
    if state.energy < 45 and EAT in valid_actions:
        return EAT

    # ---- HUNT with spear (very valuable: raw_meat → cook → +15 energy) ----
    if etype == ANIMAL and HUNT in valid_actions:
        if inv.spear and (entity.danger_level <= 2 or state.health > 50):
            return HUNT
        if entity.tool_required == TOOL_NONE and entity.danger_level <= 1:
            return HUNT   # safe bare-hands hunt
        if entity.danger_level >= 3:
            return FLEE

    # ---- GATHER: prioritise materials for missing tools, then food ----
    if etype in (RESOURCE, CRAFT_OPP, EVENT) and GATHER in valid_actions:
        # High priority: materials we still need for tools
        need_wood  = (not inv.fire and inv.wood < 1) or (not inv.spear and inv.wood < 1)
        need_stone = (not inv.spear and inv.stone < 1)
        if entity.inventory_key == "wood" and need_wood:
            return GATHER
        if entity.inventory_key == "stone" and need_stone:
            return GATHER
        # Always gather food/useful resources
        return GATHER

    # ---- CRAFT secondary tools (mid/late game) ----
    if state.turn > 6 and CRAFT_SHELTER in valid_actions:
        return CRAFT_SHELTER
    if state.turn > 8 and CRAFT_ROD in valid_actions:
        return CRAFT_ROD

    # ---- EAT proactively to keep energy buffer ----
    if state.energy < 70 and EAT in valid_actions:
        return EAT

    # ---- Flee remaining dangerous animals ----
    if etype == ANIMAL and entity.danger_level >= 2:
        return FLEE

    # ---- Hunt low-danger animals even without spear ----
    if etype == ANIMAL and HUNT in valid_actions and entity.danger_level <= 1:
        return HUNT

    return REST


def blind_policy(entity, state, valid_actions):
    """
    Policy that IGNORES the encounter (simulates Receiver without Sender).
    Can only use inventory actions + generic flee/rest.
    """
    inv = state.inventory

    # Can't see the entity, so can't address it specifically.
    # Just do inventory management.
    if state.energy < 20 and EAT in valid_actions:
        return EAT

    # Craft if possible
    if CRAFT_FIRE in valid_actions:
        return CRAFT_FIRE
    if CRAFT_SPEAR in valid_actions:
        return CRAFT_SPEAR

    if state.energy < 50 and EAT in valid_actions:
        return EAT

    if CRAFT_SHELTER in valid_actions:
        return CRAFT_SHELTER
    if CRAFT_ROD in valid_actions:
        return CRAFT_ROD

    # Can't see what's out there - conserve energy (REST > FLEE)
    # REST gives +5 energy; FLEE costs -5. Blind agent can't address
    # encounters anyway, so resting is the energy-optimal default.
    return REST


# ============================================================================
# SECTION 9: STATISTICS & ANALYSIS
# ============================================================================

def run_batch(policy_fn, policy_name: str, n_episodes: int,
              max_turns: int = 20,
              verbose: bool = False, seed: int = None,
              add_completion_bonus: bool = False) -> Dict:
    """Run n_episodes, collect aggregate statistics."""
    if seed is not None:
        random.seed(seed)

    results = {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "survived": 0,
        "total_rewards": [],
        "episode_lengths": [],
        "death_causes": defaultdict(int),
        "action_counts": defaultdict(int),
        "tools_crafted": defaultdict(int),
        "total_hunts": 0,
        "successful_hunts": 0,
        "total_chains": 0,
        "total_food_eaten": 0,
        "total_dangers_faced": 0,
        "total_dangers_mitigated": 0,
    }

    for ep in range(n_episodes):
        gs = run_episode(
            policy_fn,
            max_turns,
            verbose=(verbose and ep < 3),
            add_completion_bonus=add_completion_bonus,
        )
        results["total_rewards"].append(gs.total_reward)
        results["episode_lengths"].append(gs.turn)
        if gs.alive:
            results["survived"] += 1
        else:
            results["death_causes"][gs.death_cause] += 1
        for a, c in gs.actions_taken.items():
            results["action_counts"][a] += c
        for t in gs.tools_crafted:
            results["tools_crafted"][t] += 1
        results["total_hunts"] += gs.hunts_attempted
        results["successful_hunts"] += gs.hunts_successful
        results["total_chains"] += gs.chains_completed
        results["total_food_eaten"] += gs.food_eaten
        results["total_dangers_faced"] += gs.dangers_faced
        results["total_dangers_mitigated"] += gs.dangers_mitigated

    return results


def print_results(r: Dict):
    """Pretty-print batch results."""
    n = r["n_episodes"]
    rewards = r["total_rewards"]
    avg_r = sum(rewards) / n
    min_r = min(rewards)
    max_r = max(rewards)
    avg_len = sum(r["episode_lengths"]) / n
    surv = r["survived"] / n * 100

    print(f"\n{'='*60}")
    print(f"  Policy: {r['policy']}  ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Survival rate:    {surv:5.1f}%  ({r['survived']}/{n})")
    print(f"  Avg ep. length:   {avg_len:5.1f} / 20 turns")
    print(f"  Avg total reward: {avg_r:+8.1f}")
    print(f"  Reward range:     [{min_r:+.0f}, {max_r:+.0f}]")

    if r["death_causes"]:
        print(f"\n  Death causes:")
        for cause, count in sorted(r["death_causes"].items(),
                                    key=lambda x: -x[1]):
            pct = count / (n - r["survived"]) * 100 if n > r["survived"] else 0
            print(f"    {cause:>12}: {count:>4} ({pct:5.1f}% of deaths)")

    total_actions = sum(r["action_counts"].values())
    if total_actions > 0:
        print(f"\n  Action distribution:")
        for a_id in range(N_ACTIONS):
            count = r["action_counts"].get(a_id, 0)
            pct = count / total_actions * 100
            bar = "█" * int(pct / 2)
            print(f"    {ACTION_NAMES[a_id]:>14}: {count:>5} ({pct:4.1f}%) {bar}")

    if r["tools_crafted"]:
        print(f"\n  Tools crafted (total across episodes):")
        for tool, count in sorted(r["tools_crafted"].items(), key=lambda x: -x[1]):
            pct = count / n * 100
            print(f"    {tool:>12}: {count:>4} ({pct:5.1f}% of episodes)")

    hunts = r["total_hunts"]
    succ  = r["successful_hunts"]
    if hunts > 0:
        print(f"\n  Hunting: {succ}/{hunts} successful ({succ/hunts*100:.1f}%)")

    print(f"  Transformation chains completed: {r['total_chains']}")
    print(f"  Total food eaten: {r['total_food_eaten']}")
    if r["total_dangers_faced"] > 0:
        mit_rate = r["total_dangers_mitigated"] / r["total_dangers_faced"] * 100
        print(f"  Dangers: {r['total_dangers_mitigated']}/{r['total_dangers_faced']}"
              f" mitigated ({mit_rate:.1f}%)")
    print()


# ============================================================================
# SECTION 10: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Survival Game - Phase 2 Prototype Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python prototype.py                          # 500 eps per policy
              python prototype.py --episodes 2000          # tighter stats
              python prototype.py --episodes 3 --verbose   # detailed turn log
              python prototype.py --seed 42                # reproducible
        """))
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episodes per policy (default: 500)")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="Turns per episode (default: 20)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Print turn-by-turn log (first 3 episodes)")
    parser.add_argument(
        "--legacy_completion_bonus",
        action="store_true",
        help="Enable old +150+efficiency survival bonus (default: off for parity with training/eval)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SURVIVAL GAME - PHASE 2 PROTOTYPE")
    print("  Validating game mechanics before EGG integration")
    print("=" * 60)
    print(f"\n  Config: {args.episodes} episodes, {args.max_turns} turns, seed={args.seed}")
    print(f"  Entity catalog: {len(ALL_ENTITIES)} entities "
          f"({len(ANIMALS)} animals, {len(RESOURCES)} resources, "
          f"{len(DANGERS)} dangers, {len(CRAFT_OPPS)} craft opps, "
          f"{len(EVENTS)} events)")
    print(f"  Action space: {N_ACTIONS} actions")
    print(f"  Vector dims: {VECTOR_DIM} × {VALUES_PER_DIM} values each")

    policies = [
        (random_policy,  "RANDOM"),
        (greedy_policy,  "GREEDY"),
        (optimal_policy, "OPTIMAL"),
        (blind_policy,   "BLIND (no sender info)"),
    ]

    all_results = []
    for fn, name in policies:
        seed = args.seed
        r = run_batch(fn, name, args.episodes, args.max_turns,
                      args.verbose, seed,
                      add_completion_bonus=args.legacy_completion_bonus)
        print_results(r)
        all_results.append(r)

    # ---- Comparison summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY COMPARISON")
    print("=" * 60)
    print(f"  {'Policy':<25} {'Survival%':>10} {'AvgReward':>10} {'AvgLength':>10}")
    print(f"  {'-'*55}")
    for r in all_results:
        surv = r["survived"] / r["n_episodes"] * 100
        avg_r = sum(r["total_rewards"]) / r["n_episodes"]
        avg_l = sum(r["episode_lengths"]) / r["n_episodes"]
        print(f"  {r['policy']:<25} {surv:>9.1f}% {avg_r:>+10.1f} {avg_l:>10.1f}")

    # ---- Communication necessity ----
    opt_surv = all_results[2]["survived"] / all_results[2]["n_episodes"] * 100
    blind_surv = all_results[3]["survived"] / all_results[3]["n_episodes"] * 100
    gap = opt_surv - blind_surv
    print(f"\n  Communication necessity gap: {gap:+.1f}% "
          f"(OPTIMAL {opt_surv:.1f}% vs BLIND {blind_surv:.1f}%)")
    print(f"  → Sender information adds ~{gap:.0f}% survival advantage")

    # ---- Balance check ----
    rand_surv = all_results[0]["survived"] / all_results[0]["n_episodes"] * 100
    print(f"\n  Balance check:")
    print(f"    Random survival: {rand_surv:.1f}%  ", end="")
    if rand_surv < 5:
        print("Very hard (might be too punishing)")
    elif rand_surv < 20:
        print("Good difficulty (random rarely survives)")
    elif rand_surv < 50:
        print("Moderate (random survives too often?)")
    else:
        print("Too easy (random shouldn't survive this much)")

    print(f"    Optimal survival: {opt_surv:.1f}%  ", end="")
    if opt_surv > 90:
        print("Achievable with good strategy")
    elif opt_surv > 70:
        print("Challenging but winnable")
    elif opt_surv > 50:
        print("Quite hard (might need reward tuning)")
    else:
        print("Too hard (smart agent can't survive - tune mechanics)")

    print()


if __name__ == "__main__":
    main()
