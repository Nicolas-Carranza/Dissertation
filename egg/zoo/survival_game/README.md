# Survival Game - Game Settings & Design Document

**Author:** Nicolas Carranza Arauna  
**Date:** February 2026  
**Status:** EGG integration complete — training operational

---

## Overview

A two-agent cooperative survival game designed for studying emergent communication.
One agent (Sender) observes the environment; the other (Receiver) chooses actions.
Communication between them must emerge through training - no pre-defined language.

The game runs for a fixed number of turns. Each turn, an encounter is generated
(animal, resource, danger, crafting opportunity, or event). The Sender sees what
it is; the Receiver must decide what to do based on the Sender's message and the
shared game state.

**Goal:** Survive all turns without energy or health reaching zero.

---

## 1. Vector Schema

All entities are represented as **6-dimensional integer vectors**, where every
dimension has exactly **5 values (0–4)**. This gives 5^6 = 15,625 possible
combinations, of which ~40 are valid entities.

### Dimension 0 - Entity Type

| Value | Type     | Description                                   |
|-------|----------|-----------------------------------------------|
| 0     | Animal   | Living creature, can be hunted for food        |
| 1     | Resource | Gatherable item (food, material, water)        |
| 2     | Danger   | Environmental threat, must be addressed        |
| 3     | CraftOpp | Crafting opportunity (good spot / bonus mats)  |
| 4     | Event    | Special one-off encounter (bonuses/surprises)  |

### Dimension 1 - Subtype (5 per entity type)

| Entity Type | 0        | 1         | 2       | 3       | 4          |
|-------------|----------|-----------|---------|---------|------------|
| Animal      | predator | herbivore | reptile | aquatic | small_game |
| Resource    | raw_food | material  | water   | plant   | mineral    |
| Danger      | storm    | cold      | hazard  | terrain | disease    |
| CraftOpp    | weapon   | fire      | shelter | fishing | medicine   |
| Event       | cache    | river     | migration | bounty | cave     |

### Dimension 2 - Danger Level

| Value | Level   | Base Injury Prob | Unmitigated HP | Unmitigated Energy |
|-------|---------|------------------|----------------|--------------------|
| 0     | safe    | 0%               | 0              | 0                  |
| 1     | low     | 10%              | 5              | 3                  |
| 2     | medium  | 25%              | 12             | 6                  |
| 3     | high    | 45%              | 20             | 10                 |
| 4     | extreme | 70%              | 30             | 15                 |

### Dimension 3 - Energy Value

| Value | Category    | Actual Energy Delta |
|-------|-------------|---------------------|
| 0     | major_drain | -8                  |
| 1     | minor_drain | -3                  |
| 2     | neutral     | 0                   |
| 3     | low_gain    | +3                  |
| 4     | high_gain   | +7                  |

*Note: This dimension describes the entity's inherent energy property. Actual
energy gained from eating food uses different, higher values (see Food Energy below).*

### Dimension 4 - Tool Required

| Value | Tool        | Craft Cost         |
|-------|-------------|--------------------|
| 0     | none        | -                  |
| 1     | spear       | 1 wood + 1 stone   |
| 2     | fishing_rod | 1 wood             |
| 3     | fire        | 1 wood             |
| 4     | shelter     | 2 wood             |

### Dimension 5 - Weather Dependency

| Value | Weather  | Spawns When            |
|-------|----------|------------------------|
| 0     | any      | All weather states     |
| 1     | sunny    | Calm or Sunny          |
| 2     | rain     | Rain or Stormy         |
| 3     | calm     | Calm only              |
| 4     | night    | Night only             |

---

## 2. Entity Catalog (40 entities)

### Animals (11) - `[0, subtype, danger, energy, tool, weather]`

| Name     | Vector         | Danger | Tool Needed | Yields         | Weather |
|----------|----------------|--------|-------------|----------------|---------|
| Lion     | [0,0,4,4,1,0] | extreme| spear       | 2 raw_meat     | any     |
| Wolf     | [0,0,3,3,1,0] | high   | spear       | 1 raw_meat     | any     |
| Bear     | [0,0,4,4,1,3] | extreme| spear       | 2 raw_meat     | calm    |
| Deer     | [0,1,1,4,1,1] | low    | spear       | 2 raw_meat     | sunny   |
| Goat     | [0,1,1,3,1,0] | low    | spear       | 1 raw_meat     | any     |
| Snake    | [0,2,3,2,0,0] | high   | none        | 1 raw_meat     | any     |
| Lizard   | [0,2,1,3,0,1] | low    | none        | 1 raw_meat     | sunny   |
| Fish     | [0,3,0,3,2,2] | safe   | fishing_rod | 1 fish         | rain    |
| Salmon   | [0,3,0,4,2,2] | safe   | fishing_rod | 2 fish         | rain    |
| Rabbit   | [0,4,1,3,1,1] | low    | spear       | 1 raw_meat     | sunny   |
| Squirrel | [0,4,0,2,0,1] | safe   | none        | 1 raw_meat     | sunny   |

### Resources (9) - `[1, subtype, danger, energy, tool, weather]`

| Name        | Vector         | Danger | Yields    | Weather |
|-------------|----------------|--------|-----------|---------|
| Berries     | [1,3,1,3,0,1] | low    | 2 berries | sunny   |
| Mushrooms   | [1,3,2,3,0,0] | medium | 1 berries | any     |
| Herbs       | [1,3,0,2,0,1] | safe   | 2 herbs   | sunny   |
| Wood        | [1,1,1,2,0,1] | low    | 2 wood    | sunny   |
| Firewood    | [1,1,0,2,0,0] | safe   | 1 wood    | any     |
| Stone       | [1,4,0,2,0,0] | safe   | 2 stone   | any     |
| Flint       | [1,4,1,2,0,0] | low    | 1 stone   | any     |
| Water       | [1,2,0,3,0,0] | safe   | 2 water   | any     |
| Muddy Water | [1,2,2,3,0,2] | medium | 1 water   | rain    |

### Dangers (10) - `[2, subtype, danger, energy, tool_needed, weather]`

| Name         | Vector         | Danger  | Tool Needed | Weather |
|--------------|----------------|---------|-------------|---------|
| Storm        | [2,0,3,0,4,2] | high    | shelter     | rain    |
| Blizzard     | [2,0,4,0,4,0] | extreme | shelter     | any     |
| Cold Night   | [2,1,2,1,3,4] | medium  | fire        | night   |
| Frost        | [2,1,3,0,3,4] | high    | fire        | night   |
| Poison Plant | [2,2,3,0,0,0] | high    | none*       | any     |
| Thorns       | [2,2,2,1,0,1] | medium  | none*       | sunny   |
| Cliff        | [2,3,3,1,0,0] | high    | none*       | any     |
| Quicksand    | [2,3,4,0,0,2] | extreme | none*       | rain    |
| Fever        | [2,4,2,1,0,0] | medium  | none*       | any     |
| Infection    | [2,4,3,0,0,0] | high    | none*       | any     |

*\*`tool_required=0` means no tool can mitigate this danger - only ENDURE or FLEE.*

### Crafting Opportunities (5) - `[3, subtype, danger, energy, tool, weather]`

| Name          | Vector         | Yields   | Weather |
|---------------|----------------|----------|---------|
| Weapon Cache  | [3,0,0,2,0,0] | 1 stone  | any     |
| Dry Campsite  | [3,1,0,2,0,1] | 1 wood   | sunny   |
| Rocky Outcrop | [3,2,0,2,0,3] | 2 wood   | calm    |
| Riverbank     | [3,3,0,2,0,2] | 1 wood   | rain    |
| Herb Garden   | [3,4,0,2,0,1] | 2 herbs  | sunny   |

### Events (5) - `[4, subtype, danger, energy, tool, weather]`

| Name             | Vector         | Yields     | Weather |
|------------------|----------------|------------|---------|
| Supply Cache     | [4,0,0,3,0,0] | 2 wood     | any     |
| River Crossing   | [4,1,1,2,0,0] | 2 water    | any     |
| Animal Migration | [4,2,0,3,1,0] | 1 raw_meat | any     |
| Berry Patch      | [4,3,0,4,0,1] | 3 berries  | sunny   |
| Cave             | [4,4,0,2,0,4] | 1 wood     | night   |

---

## 3. Game State

### Starting Conditions

| Stat     | Start | Range   | Death Condition |
|----------|-------|---------|-----------------|
| Energy   | 100   | 0–100   | ≤ 0 → starvation |
| Health   | 100   | 0–100   | ≤ 0 → injury death |
| Weather  | sunny | 0–4     | -               |
| Turns    | 20    | -       | -               |

### Per-Turn Metabolism

- **Energy drain:** 2–4 per turn (random, uniform)
- Over 20 turns, total drain averages **60 energy** (range 40–80)

### Weather System

- 5 states: calm (0), sunny (1), rain (2), stormy (3), night (4)
- Changes every 4–6 turns (random interval)
- Weather determines which entities can spawn

---

## 4. Inventory System (Joint / Shared)

### Tools (boolean - once crafted, permanent)

| Tool        | Craft Cost       | Purpose                          |
|-------------|------------------|----------------------------------|
| Spear       | 1 wood + 1 stone | Hunt animals efficiently         |
| Fire        | 1 wood           | Mitigate cold, auto-cook meat    |
| Shelter     | 2 wood           | Mitigate storms                  |
| Fishing Rod | 1 wood           | Hunt aquatic animals (fish)      |

### Materials (capped)

| Material | Max Capacity |
|----------|-------------|
| Wood     | 5           |
| Stone    | 3           |

### Food & Consumables (capped at 3 each)

| Food Type   | Max | Energy When Eaten | Notes                         |
|-------------|-----|-------------------|-------------------------------|
| cooked_meat | 3   | +15               | Best food. Auto-cooked if fire exists |
| fish        | 3   | +10               | Second best                   |
| berries     | 3   | +6                | Gathered from plants          |
| water       | 3   | +5                | Gathered from water sources   |
| raw_meat    | 3   | +5                | 30% parasite risk (−12 HP)    |
| herbs       | 3   | +3                | Lowest value                  |

### Auto-Cooking (Transformation Chain)

When the EAT action is used and the inventory has both **fire** and **raw_meat**:
1. One raw_meat is automatically converted to cooked_meat
2. The cooked_meat is then eaten (+15 energy instead of +5)
3. This counts as a **completed transformation chain** (tracked statistic)
4. +5 bonus reward for eating cooked meat

---

## 5. Action Space (11 actions)

### Full Action Table

| ID | Action        | Valid When                                      | Energy Cost | Notes                           |
|----|---------------|-------------------------------------------------|-------------|----------------------------------|
| 0  | HUNT          | Encounter is Animal                             | −8 (success), −12 (fail) | Success rate depends on tools |
| 1  | GATHER        | Encounter is Resource, CraftOpp, or Event       | −2          | Adds items to inventory          |
| 2  | FLEE          | Always                                          | −5          | Safe escape, no encounter effect |
| 3  | REST          | Always                                          | +3          | Small energy recovery            |
| 4  | MITIGATE      | Encounter is Danger + have correct tool + tool≠none | −2 HP, −2 energy | Best way to handle dangers   |
| 5  | ENDURE        | Encounter is Danger                             | 60% of full damage | No tool needed              |
| 6  | EAT           | Have food in inventory                          | +3 to +15   | Eats best food first             |
| 7  | CRAFT_SPEAR   | wood ≥ 1, stone ≥ 1, don't have spear           | −3          | +50 discovery bonus              |
| 8  | CRAFT_FIRE    | wood ≥ 1, don't have fire                       | −3          | +50 discovery bonus              |
| 9  | CRAFT_SHELTER | wood ≥ 2, don't have shelter                    | −3          | +50 discovery bonus              |
| 10 | CRAFT_ROD     | wood ≥ 1, don't have fishing_rod                | −3          | +50 discovery bonus              |

### Action Validity

- FLEE and REST are **always valid**
- Other actions require preconditions (encounter type + inventory state)
- Invalid actions receive **−10 penalty** and the encounter's effects still apply
- Typical turn: 4–6 valid actions out of 11

### Hunt Success Rates

| Condition                              | Base Success Rate |
|----------------------------------------|-------------------|
| Has correct tool + danger ≤ 1          | 85%               |
| Has correct tool + danger > 1          | 65%               |
| Entity needs no tool (bare hands)      | 50%               |
| Needs tool but doesn't have it         | 15%               |

Final rate = base × (health / 100) × (min(energy, 50) / 50)

---

## 6. Encounter Generation

Each turn, one **target** entity is generated.

### Steps:

1. **Weather filter** - remove all entities incompatible with current weather
2. **Group by type** - pool remaining entities by entity type (Animal, Resource, etc.)
3. **Weighted type roll** - randomly select which *type* spawns using spawn weights
4. **Pick target** - random choice from that type's filtered pool

### Spawn Weights (balance lever)

| Entity Type | Weight | Effective Probability |
|-------------|--------|-----------------------|
| Animal      | 25     | 25%                   |
| Resource    | 35     | 35%                   |
| Danger      | 20     | 20%                   |
| CraftOpp    | 10     | 10%                   |
| Event       | 10     | 10%                   |

*Weights are relative - changing Resource from 35→50 increases its share.*

---

## 7. Unaddressed Encounters

If the agent's action does **not directly interact** with the encounter, consequences apply:

### What counts as "addressing" an encounter:

| Encounter Type         | Addressing Actions               |
|------------------------|----------------------------------|
| Animal                 | HUNT or FLEE                     |
| Resource / CraftOpp / Event | GATHER or FLEE            |
| Danger                 | MITIGATE, ENDURE, or FLEE        |

### Consequences of NOT addressing:

| Scenario                            | Consequence                            |
|-------------------------------------|----------------------------------------|
| Danger (any level), not addressed   | Full unmitigated damage (HP + energy)  |
| Animal with danger ≥ 3, not addressed | Animal attacks: 5 + danger×7 HP damage |
| Animal with danger < 3, not addressed | Nothing happens                       |
| Resource/CraftOpp/Event, not addressed | Opportunity lost, no penalty         |

**Example:** If you CRAFT_FIRE while facing a Lion (danger=4), you craft the fire
but the lion also attacks you for 33 HP damage.

---

## 8. Reward Structure

### Per-Turn Rewards:

| Component                 | Value             |
|---------------------------|-------------------|
| Alive bonus (every turn)  | +10               |
| Energy > 70 bonus         | +5                |
| Health > 70 bonus         | +3                |
| Action reward              | varies by action  |
| Unaddressed penalty       | varies by danger  |

### End-of-Episode:

| Condition           | Reward      |
|---------------------|-------------|
| Survived all turns  | +150        |
| Efficiency bonus    | +(energy + health) / 2 |
| Death by starvation | −50         |
| Death by injury     | −300        |

### Notable Action Rewards:

| Action               | Reward                           |
|----------------------|----------------------------------|
| Hunt success         | +15 + danger_level × 10          |
| Hunt failure         | −5 (+ injury risk)               |
| Gather               | +5 + items_added × 2             |
| Craft (any tool)     | +50 (discovery bonus)            |
| Mitigate danger      | +20 + danger_level × 5           |
| Flee (wise - from danger ≥ 3) | +2                      |
| Flee (from low danger)| −2                              |
| Rest                 | −1                               |
| Eat cooked meat      | +energy_gained + 5 (chain bonus) |
| Invalid action       | −10                              |

---

## 9. State Vectors (for Neural Network Input)

### GameState Vector (16 floats, all normalized 0–1):

| Index | Value                    | Normalization |
|-------|--------------------------|---------------|
| 0     | Energy                   | / 100         |
| 1     | Health                   | / 100         |
| 2     | Weather                  | / 4           |
| 3     | Turn progress            | / max_turns   |
| 4     | Has spear (bool)         | 0 or 1        |
| 5     | Has fire (bool)          | 0 or 1        |
| 6     | Has shelter (bool)       | 0 or 1        |
| 7     | Has fishing_rod (bool)   | 0 or 1        |
| 8     | Wood count               | / 5           |
| 9     | Stone count              | / 3           |
| 10    | Raw meat count           | / 3           |
| 11    | Cooked meat count        | / 3           |
| 12    | Berries count            | / 3           |
| 13    | Fish count               | / 3           |
| 14    | Water count              | / 3           |
| 15    | Herbs count              | / 3           |

*These vectors are prepared for Phase 3 (EGG integration) where neural network
agents need numeric tensor inputs. Not used by the prototype's heuristic policies.*

### Entity Vector (6 integers, each 0–4):

`[entity_type, subtype, danger_level, energy_value, tool_required, weather_dependency]`

*In EGG: the Sender receives the entity vector; the Receiver does not.*

---

## 10. Communication Necessity

The core motivation for this game design: **the Sender knows something the Receiver doesn't.**

- **Sender sees:** entity vector (what the encounter is)
- **Receiver sees:** game state vector only (energy, health, inventory, weather)
- **Communication:** Sender must encode encounter info into a discrete message

### Measured Gap (500 episodes, seed=42):

| Policy                   | Survival | Avg Reward |
|--------------------------|----------|------------|
| RANDOM                   | 18.8%   | +133.6     |
| GREEDY (full info)       | 59.2%   | +479.5     |
| OPTIMAL (full info)      | 62.6%   | +446.4     |
| BLIND (no sender info)   | 19.0%   | −61.5      |

**Communication necessity gap: 43.6%** (OPTIMAL vs BLIND)

The Sender's information adds ~44% survival advantage, confirming that
emergent communication is essential for good performance.

---

## 11. Transformation Chains

Multi-step strategies that the game rewards:

1. **Gather wood → Craft fire → Hunt animal → Auto-cook raw_meat → Eat cooked_meat (+15 energy)**
   - This is the highest-value food chain in the game
2. **Gather wood + stone → Craft spear → Hunt with spear (85% success vs 15%)**
   - Spear dramatically improves hunting outcomes
3. **Gather wood → Craft shelter → Mitigate storm (+20 reward, only 2 damage)**
   - Without shelter, a storm deals 20 HP + 10 energy
4. **Gather wood → Craft rod → Hunt fish (+10 energy, safe)**
   - Fish are danger=0 and weather-gated to rain

Each completed chain is tracked as `chains_completed` in statistics.

---

## 12. Key Balance Levers

Parameters you can tweak to adjust difficulty:

| Parameter              | Current   | Effect of Increasing              |
|------------------------|-----------|-----------------------------------|
| Starting energy        | 100       | Easier early game                 |
| Metabolic drain        | 2–4/turn  | Harder survival, more eating needed |
| Flee energy cost       | 5         | Penalizes avoidance strategy      |
| Rest energy gain       | 3         | Easier survival for passive play  |
| Food energy values     | 3–15      | Higher = easier (less eating needed) |
| Hunt success rates     | 15–85%    | Higher = more reliable food source |
| Craft discovery bonus  | 50        | Higher = stronger tool incentive  |
| Spawn weight: Danger   | 20        | Higher = more threats             |
| Spawn weight: Resource | 35        | Higher = more gathering opportunities |
| Death penalty (injury) | −300      | Higher = stronger health priority |
| Death penalty (starve) | −50       | Higher = stronger energy priority |
| Completion bonus       | +150      | Higher = stronger survival incentive |
| Max turns              | 20        | More turns = harder to survive    |
| Parasite chance (raw)  | 30%       | Higher = stronger cooking incentive |

---

## 13. Running the Prototype

```bash
# Default: 500 episodes per policy
python3 prototype.py

# More episodes for tighter statistics
python3 prototype.py --episodes 2000

# Verbose turn-by-turn log (first 3 episodes)
python3 prototype.py --episodes 5 --verbose

# Reproducible results
python3 prototype.py --seed 42

# All options
python3 prototype.py --episodes 1000 --max_turns 25 --seed 42 --verbose
```

---

## 14. File Structure

```
egg/zoo/survival_game/
├── __init__.py       # Package init
├── prototype.py      # Phase 2: Standalone simulator (no EGG dependency)
├── train.py          # Main entry point (CLI + training loop)
├── data.py           # Episode generator → PyTorch DataLoader
├── archs.py          # MLP Sender / Receiver core architectures
├── games.py          # Reinforce wiring (SenderReceiverRnnReinforce)
├── losses.py         # Reward → loss conversion + per-sample metrics
├── callbacks.py      # SurvivalGameEvaluator + MessageAnalyzer
└── README.md         # This file
```

---

## 15. Training (EGG Integration)

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Minimal smoke test (2 epochs, small data)
python -m egg.zoo.survival_game.train \
    --n_epochs 2 --batch_size 8 --vocab_size 50 --max_len 6 \
    --n_train_episodes 50 --n_val_episodes 10 --eval_freq 2 --analyze_freq 0

# Full training run (recommended defaults)
python -m egg.zoo.survival_game.train \
    --n_epochs 50 --batch_size 64 --vocab_size 50 --max_len 6 \
    --sender_hidden 128 --receiver_hidden 128 \
    --sender_entropy_coeff 0.1 --receiver_entropy_coeff 0.05 \
    --lr 1e-3 --n_train_episodes 2000 --n_val_episodes 200 \
    --eval_freq 5 --analyze_freq 10

# All options
python -m egg.zoo.survival_game.train --help
```

### Architecture

| Component | Description |
|-----------|-------------|
| **Sender core** | MLP: 30 → hidden → hidden (maps one-hot entity to RNN init state) |
| **Sender RNN** | `RnnSenderReinforce` — LSTM generates variable-length discrete messages |
| **Receiver RNN** | `RnnReceiverDeterministic` — LSTM encodes message into hidden state |
| **Receiver core** | MLP: (rnn_hidden + 27) → hidden → 11 action logits (masked by valid actions) |
| **Game wrapper** | `SenderReceiverRnnReinforce` with `CommunicationRnnReinforce` mechanics |
| **Training** | REINFORCE with mean baseline, entropy regularisation, length cost |

### Data Pipeline

Each training sample is a single encounter turn extracted from a full simulated episode:

| Tensor | Shape | Content |
|--------|-------|---------|
| `sender_input` | (30,) | One-hot entity vector (6 dims × 5 values) |
| `receiver_input` | (27,) | Game state (16) + valid action mask (11) |
| `labels` | (26,) | Packed entity vector + state info for loss computation |

Episodes are pre-generated with a random policy to ensure diverse game states.
Default: 2000 train episodes × ~16 surviving turns ≈ 32,000 training samples.

### Loss & Reward

The loss function (`SurvivalLoss`):
1. Takes the Receiver's argmax action from logits
2. Reconstructs the Entity and GameState from the labels tensor
3. Simulates the action using the prototype's resolver functions
4. Returns `loss = -reward × reward_scale` (per-sample, shape `(batch_size,)`)

Reward includes: action outcome + unaddressed encounter penalty + alive bonus (+10/turn) + health/energy bonuses.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--vocab_size` | 10 | Message vocabulary size (try 50) |
| `--max_len` | 1 | Max message length (try 6) |
| `--sender_hidden` | 128 | Sender MLP + RNN hidden size |
| `--receiver_hidden` | 128 | Receiver MLP + RNN hidden size |
| `--sender_embedding` | 32 | Sender RNN embedding dim |
| `--receiver_embedding` | 32 | Receiver RNN embedding dim |
| `--sender_cell` | lstm | RNN cell: rnn, gru, lstm |
| `--receiver_cell` | lstm | RNN cell: rnn, gru, lstm |
| `--sender_entropy_coeff` | 0.1 | Sender entropy bonus for exploration |
| `--receiver_entropy_coeff` | 0.05 | Receiver entropy bonus |
| `--reward_scale` | 0.01 | Reward scaling for loss |
| `--n_train_episodes` | 2000 | Training episodes to generate |
| `--n_val_episodes` | 200 | Validation episodes to generate |
| `--max_turns` | 20 | Turns per episode |
| `--data_seed` | 42 | Seed for reproducible data |
| `--eval_freq` | 5 | Full-episode evaluation every N epochs (0=off) |
| `--eval_episodes` | 100 | Episodes per evaluation |
| `--analyze_freq` | 10 | Message analysis every N epochs (0=off) |
| `--checkpoint_dir` | None | Directory for saving checkpoints |
| `--tensorboard` | off | Enable TensorBoard logging |

### Logged Metrics

Printed as JSON each epoch by `ConsoleLogger`:

| Metric | Meaning |
|--------|---------|
| `loss` | Mean loss (= -reward × scale) |
| `mean_reward` | Mean per-turn reward |
| `valid_action_rate` | Fraction of valid actions chosen |
| `hunt_rate` ... `craft_rate` | Action distribution |
| `sender_entropy` | Sender's message entropy (exploration) |
| `length` | Mean message length |

The `SurvivalGameEvaluator` callback runs full sequential episodes every `eval_freq` epochs, reporting true survival rate and action distribution.


## Appendix A: Parameter Definitions

### A.1 Training Mode Parameters

| Parameter | CLI Flag | Type | Description |
|-----------|----------|------|-------------|
| **mode** | `--mode` | str | Training paradigm. `rf` = REINFORCE (policy gradient with sampling), `gs` = Gumbel-Softmax (differentiable relaxation, no sampling). Default: `rf` |
| **n_epochs** | `--n_epochs` | int | Total training epochs. Each epoch = one full pass through the training dataset. Default: 10 |

### A.2 Agent Architecture Parameters

| Parameter | CLI Flag | Type | Description |
|-----------|----------|------|-------------|
| **sender_hidden** | `--sender_hidden` | int | Hidden dimension of the Sender MLP (entity→embedding) and the Sender RNN (embedding→message). Default: 128 |
| **receiver_hidden** | `--receiver_hidden` | int | Hidden dimension of the Receiver RNN (message→embedding) and all downstream heads (action logits, reconstruction). Default: 128 |
| **sender_embedding** | `--sender_embedding` | int | Embedding dimension for the Sender RNN's input projection. The entity vector (6-d) is first projected to `sender_hidden`, then this becomes the RNN's hidden state; the RNN input tokens are embedded to `sender_embedding`. Default: 32 |
| **receiver_embedding** | `--receiver_embedding` | int | Embedding dimension for the Receiver RNN's input tokens (message symbols). Default: 32 |
| **sender_cell** | `--sender_cell` | str | RNN cell type for Sender: `rnn`, `gru`, or `lstm`. Default: `lstm` |
| **receiver_cell** | `--receiver_cell` | str | RNN cell type for Receiver: `rnn`, `gru`, or `lstm`. Default: `lstm` |

### A.3 Gumbel-Softmax Communication Parameters

| Parameter | CLI Flag | Type | Description |
|-----------|----------|------|-------------|
| **temperature** | `--temperature` | float | Initial temperature $\tau$ for the Gumbel-Softmax distribution used by the Sender to generate message tokens. Higher $\tau$ → softer (more uniform) token distributions; lower $\tau$ → sharper (more one-hot) tokens. At $\tau \to 0$, GS converges to argmax (discrete). Default: 2.0 |
| **temperature_decay** | `--temperature_decay` | float | Multiplicative decay applied to temperature each epoch: $\tau_{t+1} = \max(\tau_{\min}, \tau_t \times d)$. Set to 0.0 to disable decay (constant temperature). Default: 0.9 |
| **temperature_minimum** | `--temperature_minimum` | float | Floor for temperature decay. Prevents temperature from going to zero (which would kill gradients). Default: 0.1 |

### A.4 Loss Function Parameters

| Parameter | CLI Flag | Type | Description |
|-----------|----------|------|-------------|
| **reward_scale** | `--reward_scale` | float | Scaling factor $\alpha$ applied to the game reward component of the loss: $L_{\text{game}} = -\alpha \sum_a p(a) \cdot r(a)$. Controls how much the reward signal influences the total loss relative to the reconstruction loss. Default: 0.2 |
| **recon_weight** | `--recon_weight` | float | Weight $\beta$ for the entity reconstruction loss: $L_{\text{recon}} = \beta \cdot \text{CE}(\hat{e}, e)$. Higher values prioritise communication accuracy; lower values give more relative weight to reward-based learning. Only used in GS mode. Default: 2.0 |
| **action_entropy_coeff** | `--action_entropy_coeff` | float | Coefficient $\lambda$ for the action entropy bonus: $L_{\text{entropy}} = -\lambda \cdot H(\mathbf{p}_{\text{action}})$, where $H(\mathbf{p}) = -\sum_i p_i \ln p_i$. Positive $\lambda$ encourages the agent to maintain a diverse action distribution (exploration). Only used in GS mode. Added in Run 6. Default: 0.1 |
| **action_temperature** | `--action_temperature` | float | Temperature $\tau_a$ for the action softmax: $\mathbf{p}_{\text{action}} = \text{softmax}(\mathbf{z}/\tau_a)$. Higher $\tau_a$ flattens the distribution (more exploration); $\tau_a = 1.0$ is standard softmax. Only used in GS mode. Added in Run 6. Default: 2.0 |
| **reward_normalise** | `--reward_normalise` | flag | When set, normalises each sample's 11-action reward vector to 0 mean and unit variance: $r'_a = (r_a - \bar{r})/\sigma_r$. This prevents samples with uniformly high/low rewards from dominating the gradient. Only used in GS mode. Added in Run 6. Default: off |
| **sender_entropy_coeff** | `--sender_entropy_coeff` | float | Entropy regularisation for the Sender's message token distributions. RF mode only. Higher values encourage the sender to explore different messages. Default: 0.1 |
| **receiver_entropy_coeff** | `--receiver_entropy_coeff` | float | Entropy regularisation for the Receiver's action distribution. RF mode only. Default: 0.05 |

### A.5 Complete Loss Formula (GS Mode)

$$L = \underbrace{-\alpha \sum_{a=1}^{11} p_a \cdot r'_a}_{\text{game reward loss}} + \underbrace{\beta \cdot \text{CE}(\hat{e}, e)}_{\text{reconstruction loss}} + \underbrace{(-\lambda) \cdot H(\mathbf{p}_{\text{action}})}_{\text{entropy bonus (reduces loss)}}$$

where:
- $p_a = \text{softmax}(\mathbf{z}_a / \tau_a)$ — soft action probabilities
- $r'_a$ — (optionally normalised) reward for action $a$
- $\hat{e}$ — predicted entity class (40-way), $e$ — true entity index
- $H(\mathbf{p}) = -\sum_i p_i \ln p_i$ — Shannon entropy

### A.6 Data Parameters

| Parameter | CLI Flag | Type | Description |
|-----------|----------|------|-------------|
| **n_episodes** | `--n_episodes` | int | Total episodes to generate. Each episode is ~20 turns long. Episodes are split into train/val/test. Default: 10000 |
| **max_turns** | `--max_turns` | int | Maximum turns (encounters) per episode. Default: 20 |
| **train_frac** | `--train_frac` | float | Fraction of episodes for training. Default: 0.8 |
| **val_frac** | `--val_frac` | float | Fraction of episodes for validation. Default: 0.1 |
| **data_seed** | `--data_seed` | int | Random seed for episode generation, ensuring reproducibility. Default: 42 |
| **vocab_size** | `--vocab_size` | int | Number of symbols in the communication vocabulary. Default: 50 |
| **max_len** | `--max_len` | int | Maximum message length (in symbols). Messages are padded to `max_len + 1` with EOS. Default: 2 |
| **batch_size** | `--batch_size` | int | Mini-batch size for training. Default: 64 (from EGG) or 32 |
| **lr** | `--lr` | float | Learning rate for Adam optimiser. Default: 1e-3 |

---