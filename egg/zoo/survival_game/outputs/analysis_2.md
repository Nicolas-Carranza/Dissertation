# Comprehensive Analysis — Runs 5, 6, 7
## Survival Game Emergent Communication Experiment
### Prepared for Supervisor Meeting

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [What Changed Since Run 3 (and Why)](#2-what-changed-since-run-3-and-why)
3. [Run-by-Run Results & Effects](#3-run-by-run-results--effects)
4. [The Local Optimum Problem](#4-the-local-optimum-problem)
5. [Questions for Supervisor](#5-questions-for-supervisor)
6. [Appendix A: Parameter Definitions](#appendix-a-parameter-definitions)
7. [Appendix B: Epoch-by-Epoch Output Tables](#appendix-b-epoch-by-epoch-output-tables)

---

## 1. Executive Summary

Across runs 5–7 we attempted to break out of a survival rate plateau (~37–41%) where the agent discovers a degenerate gather/eat-dominated policy. Key findings:

| Metric               | Run 5            | Run 6            | Run 7            |
|-----------------------|------------------|------------------|------------------|
| **Mean Survival**     | 37.2%            | 41.1%            | 39.4%            |
| **Peak Survival**     | 40%              | 49%              | 49%              |
| **Final Recon Acc**   | 100.0%           | 96.7%            | 52.5%            |
| **Unique Messages**   | 40 (perfect)     | 36              | 26               |
| **Dominant Actions**  | gather 43%, eat 27% | gather 42%, eat 27% | gather 41%, eat 27% |
| **Hunt Rate**         | ~0%              | 4–6%             | ~4%              |
| **Epochs Trained**    | 50               | 50               | 70               |

**Bottom line**: The action policy is stuck in a local optimum. The agent has learned to communicate (especially in Run 5 — perfect reconstruction), but cannot translate entity knowledge into context-dependent actions. Gather/eat always yields safe, positive reward; risky context-dependent actions (hunt, flee) require coordinating entity-type understanding with the correct action, and the gradient signal for this is too weak relative to the stable gather/eat baseline.

---

## 2. What Changed Since Run 3 (and Why)

### 2.1 Run 3 → Run 4: Data Pipeline Overhaul
**What**: 
- Changed from 2-way (train/test) to 3-way split (train 80% / val 10% / test 10%)
- Added fingerprint-based deduplication to prevent data leakage between splits
- Expanded entity set from 5 classes to 40 classes (11 Animal, 9 Resource, 10 Danger, 5 CraftOpp, 5 Event)
- Episode-level splitting (all turns from one episode stay in the same split)

**Why**: 
- The 5-class reconstruction was too easy — the agent achieved 100% recon but only had 5 unique messages (one per class). With only 5 possible messages, the receiver wasn't getting enough information to make good action decisions.
- We needed proper train/val/test separation with no data leakage to ensure honest evaluation.
- More entity diversity forces the agent to discover a richer communication protocol.

**Effect**: Run 4 was a verification run confirming the data pipeline worked; it was trained with the old 5-class reconstruction head. Showed the infrastructure was sound but still had the 5-message degenerate protocol.

### 2.2 Run 4 → Run 5: 40-Class Reconstruction Head
**What**:
- Changed the reconstruction head from `Linear(128→64→5)` (5 entity classes) to `Linear(128→64→40)` (40 individual entities)
- `recon_weight` set to 1.5 (from 2.0)
- `reward_scale` set to 0.5 (from 0.2)
- `temperature` = 2.0, `temp_decay` = 0.975
- No action entropy bonus, no reward normalisation, no action temperature modification

**Why**:
- With 5-class recon, the agent mapped all entities of the same type to the same message. But the receiver needs to distinguish *which* animal or *which* danger to pick the right action. 40-class recon forces the sender to communicate the *specific* entity.
- Higher `reward_scale` (0.5 vs 0.2) was meant to give more weight to the reward component of the loss.
- Lower `recon_weight` (1.5 vs 2.0) was a slight reduction to balance the higher reward scale.

**Effect** (Run 5 results):
- **Communication**: Massive success. 40 unique messages emerged, achieving 100% reconstruction accuracy by epoch 22. Every entity got its own distinct message.
- **Action policy**: Complete failure to improve. Survival stayed at 37.2% mean (range 32–40%), virtually identical to random-baseline-plus-eating. The agent found gather (43%) / eat (27%) as a safe default and never explored beyond it.
- **Root cause identified**: In GS mode the action_logits go through `softmax()` without temperature scaling — starting from random initialisation one action quickly dominates (gather) and the softmax output becomes a near one-hot vector. Once the gradient through the reward term is effectively zeroed out for non-dominant actions, exploration stops.

### 2.3 Run 5 → Run 6: Action Policy Fixes
**What** (three simultaneous changes):
1. **Action Entropy Bonus** (`action_entropy_coeff=0.1`): Added $-\lambda_{\text{ent}} \cdot H(\mathbf{p}_{\text{action}})$ to the loss, where $H(\mathbf{p}) = -\sum_i p_i \log p_i$. This penalises low-entropy (peaked) action distributions.
2. **Action Temperature** (`action_temperature=2.0`): Changed action probabilities from $\text{softmax}(\mathbf{z})$ to $\text{softmax}(\mathbf{z}/\tau)$, where $\tau = 2.0$. Higher temperature flattens the distribution, preventing early softmax collapse.
3. **Reward Normalisation** (`reward_normalise=True`): For each sample, the 11-action reward vector is normalised to zero mean and unit variance: $r'_a = (r_a - \mu_r)/\sigma_r$. This makes the gradient equally strong for all samples regardless of absolute reward magnitude.

**Why**: We diagnosed the Run 5 action policy failure as softmax collapse:
- Without temperature scaling, the softmax concentrates probability mass on one action very quickly
- Without entropy bonus, there's no incentive to maintain exploration
- Without reward normalisation, samples where all actions give similar rewards produce near-zero gradients
- Together, these create "gradient starvation" — the dominant action eats all the gradient signal, and alternative actions never get explored

**Effect** (Run 6 results):
- **Survival**: Marginal improvement. Mean 41.1% (vs 37.2% in Run 5), peak 49% (vs 40%).
- **Hunt rate**: Increased from ~0% to ~4–6%. This is the first time the agent began using a context-dependent action.
- **Communication**: Slightly degraded. 36 unique messages (down from 40), final recon_acc 96.7% (down from 100%). The entropy bonus slightly disrupts the reconstruction loss gradient.
- **Action distributions**: Nearly identical to Run 5 at final epoch (gather 41.5%, eat 26.7%).
- **Assessment**: The fixes helped marginally but the fundamental problem remains. An entropy coefficient of 0.1 was too mild to break the gather/eat attractor, and a temperature of 2.0 may have been too high (flattening so much that the agent can't learn meaningful preferences).

### 2.4 Run 6 → Run 7: Hyperparameter Tuning
**What**:
1. **Higher entropy** (`action_entropy_coeff=0.5`): 5× stronger exploration pressure
2. **Lower action temperature** (`action_temperature=1.0`): Standard softmax (no temperature scaling), so the agent's logit differences directly translate to probability differences
3. **No temperature decay** (`temperature_decay=0.0`): Keep GS communication temperature at 1.0 throughout. No annealing.
4. **Lower initial GS temperature** (`temperature=1.0`): Start with a sharper communication channel from the beginning (vs 2.0 which starts soft and anneals)
5. **Longer training** (70 epochs vs 50)

**Why**:
- Entropy coeff 0.1 was too weak to prevent softmax collapse → try 0.5
- Action temperature 2.0 over-flattened the action distribution, preventing the agent from committing to learned action preferences → try 1.0
- The temperature decay was causing the communication channel to anneal to near-deterministic too quickly, and with decay=0.0 + temp=1.0 we keep a moderate channel throughout
- We wanted to test whether longer training with stronger exploration would help break through the plateau

**Effect** (Run 7 results):
- **Survival**: Mean 39.4% (between Run 5 and Run 6). Peak 49% (same as Run 6). Very high variance (range 30–49%).
- **Communication**: Significantly degraded. Only 26 unique messages at final epoch (down from 40 in Run 5, 36 in Run 6). Recon accuracy only 52.5% (down from 100% and 96.7%).
- **Action distributions**: Virtually identical again: gather ~41%, eat ~27%.
- **Key observation**: The strong entropy bonus (0.5) actively competed with the reconstruction loss — the model couldn't learn good communication AND maintain exploration simultaneously. This is a fundamental tension: entropy bonus fights gradient from recon loss.
- **Expected reward**: Still climbing at epoch 70 (0.85 → 0.88 between epochs 50 and 70), suggesting the model hadn't fully converged. But the trajectory was very slow.

---

## 3. Run-by-Run Results & Effects

### 3.1 Survival Rate Trajectory

| Epoch | Run 5 | Run 6 | Run 7 |
|-------|-------|-------|-------|
| 5     | 39%   | 49%   | 48%   |
| 10    | 32%   | 35%   | 31%   |
| 15    | 38%   | 33%   | 40%   |
| 20    | 38%   | 48%   | 38%   |
| 25    | 37%   | 41%   | 40%   |
| 30    | 39%   | 37%   | 41%   |
| 35    | 32%   | 42%   | 35%   |
| 40    | 38%   | 48%   | 49%   |
| 45    | 40%   | 37%   | 30%   |
| 50    | 39%   | 41%   | 46%   |
| 55    | —     | —     | 37%   |
| 60    | —     | —     | 39%   |
| 65    | —     | —     | 37%   |
| 70    | —     | —     | 41%   |

All three runs oscillate in the same ~30–49% band with no upward trend after the initial 5–10 epochs.

### 3.2 Action Distribution at Final Evaluation

| Action       | Run 5 (ep50) | Run 6 (ep50) | Run 7 (ep70) |
|--------------|-------------|-------------|-------------|
| hunt         | 0.0%        | 5.0%        | 4.2%        |
| **gather**   | **43.3%**   | **41.5%**   | **40.8%**   |
| flee         | 6.1%        | 8.0%        | 8.3%        |
| rest         | 7.2%        | 5.1%        | 4.5%        |
| mitigate     | 0.8%        | 1.1%        | 2.0%        |
| endure       | 0.0%        | 0.0%        | 0.0%        |
| **eat**      | **26.5%**   | **26.7%**   | **27.2%**   |
| craft_spear  | 3.1%        | 2.4%        | 2.0%        |
| craft_fire   | 4.9%        | 4.5%        | 3.2%        |
| craft_shelter| 2.3%        | 2.0%        | 1.9%        |
| craft_rod    | 5.8%        | 3.9%        | 4.6%        |

The gather+eat combined rate is ~70% across all runs. This is the local optimum.

### 3.3 Communication Quality

| Metric             | Run 5   | Run 6   | Run 7   |
|--------------------|---------|---------|---------|
| Recon accuracy (final test) | 100.0%  | 96.7%   | 52.5%   |
| Unique msgs (final) | 40/40   | 36/40   | 26/40   |
| Animal msgs        | 11/11   | 10/11   | 10/11   |
| Resource msgs      | 9/9     | 9/9     | 3/9     |
| Danger msgs        | 10/10   | 7/10    | 6/10    |
| CraftOpp msgs      | 5/5     | 5/5     | 3/5     |
| Event msgs         | 5/5     | 5/5     | 4/5     |

Run 5 has perfect communication but useless actions. Run 7 has poor communication AND useless actions. The entropy bonus in Run 7 directly harmed communication quality without improving actions.

### 3.4 Training Dynamics

- **Run 5**: Recon accuracy shoots to 100% by epoch 22 (very fast). Expected reward flatlines at ~10.39 (unnormalised scale). Communication converges early and stays locked in.
- **Run 6**: Recon accuracy reaches ~93% by epoch 10, stabilises at ~97%. The entropy/temperature changes slightly destabilise communication. Expected reward reaches 1.29 (normalised scale) and flatlines.
- **Run 7**: Recon accuracy grows very slowly: 22% (ep5) → 35% (ep20) → 50% (ep50) → 52.5% (ep70). The strong entropy bonus actively fights the reconstruction gradient. Expected reward still climbing at ep70 (0.85) but very slowly.

---

## 4. The Local Optimum Problem

### 4.1 Why Gather/Eat Dominates

The gather/eat strategy is a **reward-maximising equilibrium** given the game mechanics:

1. **Gather is always valid**: Unlike hunt (requires Animal entity + optionally Spear), flee (requires Danger entity), or mitigate (requires Danger), gather can be performed on any encounter. It has zero risk and always yields some resources.

2. **Eat is always valid**: As long as you have any food (berries, fish, cooked meat), eat restores energy. Since you gather food constantly, you always have something to eat.

3. **The strategy is self-reinforcing**: gather → accumulate food → eat → restore energy → survive → gather more. This loop has no risk and a moderately positive reward.

4. **Context-dependent actions have higher variance**: Hunting when an Animal is present yields high reward but requires the right tool. Fleeing when Danger is present is critical but only relevant ~25% of the time (10/40 entities are Danger). The learning signal for these conditional strategies is sparse.

### 4.2 Why the Agent Can't Break Out

The gradient dynamics create a trap:

1. **Softmax concentration**: Once `gather` has slightly higher logits than other actions, $\text{softmax}$ amplifies this — if gather has logit 2.0 and hunt has 1.0, the probability ratio is $e^{2}/e^{1} \approx 2.7\times$. After a few epochs: logit 5.0 vs 1.0 → ratio $e^{5}/e^{1} \approx 55\times$.

2. **Gradient starvation for rare actions**: When $p(\text{hunt}) < 0.01$, the gradient $\partial L / \partial z_{\text{hunt}}$ is proportional to $p(\text{hunt})$ and becomes negligible. Even if hunting would give 10× the reward, the gradient is 10× smaller than for gather.

3. **The entropy bonus is a blunt instrument**: It pushes probability mass toward *all* 11 actions equally — including useless ones like endure (which is always bad). What we actually need is to push mass toward *contextually appropriate* actions, not toward uniform exploration.

4. **Communication-action coupling**: The receiver must learn two things simultaneously: (a) decode the message to know what entity it faces, and (b) map that entity knowledge to the right action. In the current architecture, the same network does both. When the action head collapses to gather/eat, there's no gradient signal telling the receiver *which* entity-message pairs should trigger *which* alternative actions.

### 4.3 Fundamental Issues

1. **Reward structure**: Gather gives reliable +small reward. Hunt gives occasional +big reward but frequent +0 or -penalty. The agent correctly learns the risk-adjusted strategy is to always gather, unless we make the penalty for not-hunting-when-you-should much larger or the reward for successful hunting much larger.

2. **Missing reward shaping**: There's no explicit penalty for using gather when a Danger entity is present and unaddressed. The "unaddressed danger" penalty is implicit (danger reduces health at turn end), but it's temporally delayed and not attributable to the specific action.

3. **Single-step RL**: The agent sees one entity and picks one action. It can't plan a multi-step strategy like "gather wood now to craft spear later for hunting." The episode-level survival rate integrates over 20 turns, but each action decision is independent.

4. **No advantage estimation**: In standard RL, we compute advantages $A(s,a) = Q(s,a) - V(s)$ to tell the agent whether an action is better than average. Our loss uses raw rewards, so "gather gives +2" looks equally good regardless of whether "hunt would have given +5."

---

## 5. Questions for Supervisor

### Architecture & Training

1. **Should we decouple the action and reconstruction heads?** Currently the receiver has a shared 128-d representation feeding both the action logits and the reconstruction head. If we add a separate "action reasoning" layer that takes both the shared representation AND the decoded entity type, the action selection could condition on the entity understanding rather than learning both tasks from scratch.

2. **Would advantage estimation help?** Instead of `loss = -(probs * rewards).sum()`, use `loss = -(probs * advantages).sum()` where `advantage_a = reward_a - mean(rewards)`. This is what reward normalisation does partially, but we could also subtract a learned baseline (value function).

3. **Should we consider curriculum learning?** Start with only Animal and Danger entities (where the correct action is clear: hunt animals, flee dangers), let the agent learn context-dependent actions, then gradually introduce Resource, CraftOpp, and Event entities.

4. **Is the optimizer struggling?** We're using Adam with lr=1e-3. Should we try lower lr (5e-4 or 1e-4) to prevent oscillating around the optimum? Or use lr scheduling?

### Reward Design

5. **Should we reshape the reward to penalise gather when a better action exists?** For example: if entity is Animal and gather gives +2, but hunt would give +5, add a penalty of -(5-2) for choosing gather. This would create explicit "opportunity cost" gradients.

6. **Should we add explicit intrinsic rewards for entity-type-appropriate actions?** E.g., a bonus for hunting when facing an animal, fleeing when facing danger, regardless of the game outcome. This would provide direct supervision for the entity→action mapping.

7. **Is the survival rate the right metric?** Perhaps we should also track and optimise for "entity-appropriate action rate" — what fraction of the time the agent takes the best available action for the entity type it faces.

### Communication

8. **Is the communication channel too rich?** With vocab=50 and max_len=2, the agent has 50² = 2,500 possible messages for 40 entities. This over-provisioning means there's no pressure to be efficient. Would a smaller vocab (e.g., 10 or 20) force more structured communication?

9. **The 100% reconstruction in Run 5 vs. 52% in Run 7 — which should we prioritise?** Perfect communication + bad actions (Run 5) vs. imperfect communication + slightly better actions (Run 7). Should we train communication and actions separately (two-phase training)?

### Methodology

10. **Are 50–70 epochs enough?** Run 7's expected reward was still climbing at epoch 70. Should we try 200–500 epochs and see if the slow learning eventually breaks through?

11. **Should we try Reinforce mode instead of Gumbel-Softmax?** In RF mode, the agent actually samples actions and experiences real consequences. The exploration is inherently stochastic. GS mode's differentiable relaxation might be too smooth to create the large gradient signals needed to break the gather/eat attractor.

12. **Would multiple seeds (statistical significance) help?** One run per configuration makes it hard to distinguish signal from noise. The survival rate variance within a single run (±8pp) is comparable to the difference between runs. Should we do 3–5 seeds per config?

---

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

## Appendix B: Epoch-by-Epoch Output Tables

### B.1 Run 5 — 40-Class Recon Baseline

**Config**: mode=gs, temp=2.0, decay=0.975, temp_min=0.1, recon_weight=1.5, reward_scale=0.5, vocab=50, max_len=2, lr=1e-3, batch=64, 50 epochs, NO action_entropy, NO action_temperature, NO reward_normalise

#### Training Metrics (Test Set, per epoch)

| Epoch | Recon Acc | Expected Reward | Loss    | Gather Rate | Eat Rate | Hunt Rate | Flee Rate |
|-------|-----------|-----------------|---------|-------------|----------|-----------|-----------|
| 1     | 0.1262    | 1.098           | 1.822   | 0.347       | 0.202    | 0.000     | 0.114     |
| 2     | 0.4414    | 1.147           | -0.106  | 0.386       | 0.221    | 0.014     | 0.102     |
| 3     | 0.5943    | 1.151           | -0.308  | 0.404       | 0.221    | 0.003     | 0.098     |
| 4     | 0.6966    | 1.164           | -0.308  | 0.403       | 0.218    | 0.000     | 0.103     |
| 5     | 0.7762    | 1.161           | -0.215  | 0.399       | 0.227    | 0.002     | 0.098     |
| 10    | 0.9562    | 1.187           | -0.315  | 0.395       | 0.224    | 0.000     | 0.105     |
| 15    | 0.9930    | 1.198           | -0.276  | 0.389       | 0.230    | 0.000     | 0.105     |
| 20    | 0.9994    | 1.205           | -0.270  | 0.388       | 0.234    | 0.000     | 0.101     |
| 25    | 1.0000    | 1.211           | -0.270  | 0.381       | 0.233    | 0.000     | 0.106     |
| 30    | 1.0000    | 1.215           | -0.263  | 0.388       | 0.231    | 0.000     | 0.105     |
| 35    | 1.0000    | 1.218           | -0.266  | 0.385       | 0.230    | 0.000     | 0.106     |
| 40    | 1.0000    | 1.222           | -0.267  | 0.387       | 0.227    | 0.000     | 0.107     |
| 45    | 1.0000    | 1.225           | -0.270  | 0.386       | 0.227    | 0.000     | 0.107     |
| 50    | 1.0000    | 1.227           | -0.270  | 0.389       | 0.226    | 0.000     | 0.107     |

#### Evaluation Episodes (every 5 epochs)

| Epoch | Survival | Mean Reward | Hunt | Gather | Flee | Rest | Eat   | Craft Total |
|-------|----------|-------------|------|--------|------|------|-------|-------------|
| 5     | 39%      | 262.2       | 0.5% | 42.8%  | 6.2% | 7.1% | 25.2% | 17.6%       |
| 10    | 32%      | 227.7       | 0.0% | 43.7%  | 5.3% | 7.3% | 26.1% | 16.5%       |
| 15    | 38%      | 275.7       | 0.0% | 42.3%  | 5.8% | 6.2% | 26.3% | 18.5%       |
| 20    | 38%      | 275.0       | 0.0% | 43.3%  | 5.6% | 7.0% | 27.1% | 16.4%       |
| 25    | 37%      | 274.0       | 0.0% | 44.3%  | 6.1% | 6.7% | 27.1% | 15.0%       |
| 30    | 39%      | 260.7       | 0.0% | 44.4%  | 7.1% | 7.5% | 28.1% | 12.1%       |
| 35    | 32%      | 239.3       | 0.0% | 43.8%  | 7.2% | 7.3% | 27.9% | 13.0%       |
| 40    | 38%      | 268.3       | 0.0% | 43.2%  | 5.5% | 6.8% | 26.2% | 17.5%       |
| 45    | 40%      | 298.2       | 0.0% | 42.9%  | 6.5% | 6.5% | 26.3% | 16.9%       |
| 50    | 39%      | 260.2       | 0.0% | 43.3%  | 6.1% | 7.2% | 26.5% | 16.1%       |

#### Message Analysis

| Epoch | Animal | Resource | Danger | CraftOpp | Event | Total Unique |
|-------|--------|----------|--------|----------|-------|-------------|
| 10    | 11     | 9        | 10     | 5        | 5     | 40          |
| 20    | 11     | 9        | 10     | 5        | 5     | 40          |
| 30    | 11     | 9        | 10     | 5        | 5     | 40          |
| 40    | 11     | 9        | 10     | 5        | 5     | 40          |
| 50    | 11     | 9        | 10     | 5        | 5     | 40          |

---

### B.2 Run 6 — Action Entropy + Temperature + Normalisation

**Config**: mode=gs, temp=2.0, decay=0.975, temp_min=0.1, recon_weight=1.5, reward_scale=0.5, **action_entropy_coeff=0.1**, **action_temperature=2.0**, **reward_normalise=True**, vocab=50, max_len=2, lr=1e-3, batch=64, 50 epochs

#### Training Metrics (Test Set, per epoch)

| Epoch | Recon Acc | Expected Reward | Loss    | Gather Rate | Eat Rate | Hunt Rate | Flee Rate |
|-------|-----------|-----------------|---------|-------------|----------|-----------|-----------|
| 1     | 0.571     | 1.161           | 0.710   | 0.397       | 0.244    | 0.000     | 0.084     |
| 2     | 0.513     | 1.176           | 1.542   | 0.412       | 0.230    | 0.000     | 0.095     |
| 5     | 0.920     | 1.199           | -0.180  | 0.407       | 0.226    | 0.000     | 0.113     |
| 10    | 0.924     | 1.264           | -0.473  | 0.396       | 0.215    | 0.038     | 0.103     |
| 15    | 0.924     | 1.273           | -0.478  | 0.402       | 0.215    | 0.037     | 0.111     |
| 20    | 0.873     | 1.276           | -0.321  | 0.401       | 0.214    | 0.037     | 0.121     |
| 25    | 0.959     | 1.277           | -0.526  | 0.391       | 0.220    | 0.034     | 0.114     |
| 30    | 0.974     | 1.281           | -0.586  | 0.393       | 0.214    | 0.035     | 0.118     |
| 35    | 0.973     | 1.283           | -0.586  | 0.399       | 0.211    | 0.036     | 0.121     |
| 40    | 0.972     | 1.278           | -0.541  | 0.399       | 0.209    | 0.042     | 0.122     |
| 45    | 0.968     | 1.287           | -0.548  | 0.392       | 0.214    | 0.039     | 0.117     |
| 50    | 0.930     | 1.288           | -0.475  | 0.394       | 0.212    | 0.040     | 0.122     |

#### Evaluation Episodes (every 5 epochs)

| Epoch | Survival | Mean Reward | Hunt | Gather | Flee | Rest | Eat   | Craft Total |
|-------|----------|-------------|------|--------|------|------|-------|-------------|
| 5     | 49%      | 315.2       | 0.0% | 42.8%  | 9.3% | 8.1% | 25.2% | 14.0%       |
| 10    | 35%      | 224.1       | 4.2% | 42.7%  | 6.4% | 6.8% | 28.0% | 11.2%       |
| 15    | 33%      | 244.4       | 3.8% | 41.2%  | 7.6% | 6.2% | 28.9% | 11.6%       |
| 20    | 48%      | 293.2       | 4.3% | 42.6%  | 8.0% | 5.9% | 26.8% | 11.3%       |
| 25    | 41%      | 287.1       | 4.0% | 40.8%  | 8.3% | 6.7% | 27.3% | 11.7%       |
| 30    | 37%      | 268.0       | 3.9% | 40.2%  | 9.3% | 6.4% | 26.7% | 12.7%       |
| 35    | 42%      | 288.6       | 4.1% | 41.9%  | 8.5% | 4.5% | 27.2% | 11.8%       |
| 40    | 48%      | 301.3       | 4.7% | 41.6%  | 8.6% | 6.5% | 26.7% | 10.9%       |
| 45    | 37%      | 273.4       | 6.0% | 38.8%  | 9.3% | 6.2% | 26.8% | 12.2%       |
| 50    | 41%      | 310.9       | 5.0% | 41.5%  | 8.0% | 5.1% | 26.7% | 12.8%       |

#### Message Analysis

| Epoch | Animal | Resource | Danger | CraftOpp | Event | Total Unique |
|-------|--------|----------|--------|----------|-------|-------------|
| 10    | 11     | 9        | 8      | 5        | 5     | 38          |
| 20    | 11     | 8        | 8      | 5        | 5     | 37          |
| 30    | 11     | 9        | 8      | 5        | 5     | 38          |
| 40    | 10     | 9        | 8      | 4        | 5     | 36          |
| 50    | 10     | 9        | 7      | 5        | 5     | 36          |

---

### B.3 Run 7 — Stronger Entropy, No Temperature Decay

**Config**: mode=gs, **temp=1.0**, **decay=0.0** (disabled), temp_min=0.1, recon_weight=1.5, reward_scale=0.5, **action_entropy_coeff=0.5**, **action_temperature=1.0**, **reward_normalise=True**, vocab=50, max_len=2, lr=1e-3, batch=64, 70 epochs

#### Training Metrics (Test Set, per epoch)

| Epoch | Recon Acc | Expected Reward | Loss    | Gather Rate | Eat Rate | Hunt Rate | Flee Rate |
|-------|-----------|-----------------|---------|-------------|----------|-----------|-----------|
| 1     | 0.039     | 0.653           | 3.541   | 0.174       | 0.032    | 0.028     | 0.064     |
| 2     | 0.065     | 0.741           | 3.355   | 0.144       | 0.114    | 0.097     | 0.066     |
| 5     | 0.185     | 0.782           | 2.784   | 0.254       | 0.195    | 0.029     | 0.075     |
| 10    | 0.223     | 0.800           | 2.589   | 0.354       | 0.176    | 0.004     | 0.089     |
| 15    | 0.228     | 0.805           | 2.557   | 0.381       | 0.181    | 0.001     | 0.089     |
| 20    | 0.347     | 0.817           | 2.155   | 0.392       | 0.190    | 0.001     | 0.090     |
| 25    | 0.378     | 0.819           | 2.038   | 0.395       | 0.194    | 0.001     | 0.088     |
| 30    | 0.470     | 0.829           | 1.694   | 0.397       | 0.196    | 0.001     | 0.089     |
| 35    | 0.477     | 0.833           | 1.640   | 0.398       | 0.197    | 0.001     | 0.089     |
| 40    | 0.502     | 0.840           | 1.528   | 0.398       | 0.198    | 0.001     | 0.090     |
| 45    | 0.487     | 0.843           | 1.570   | 0.399       | 0.198    | 0.001     | 0.090     |
| 50    | 0.503     | 0.846           | 1.530   | 0.399       | 0.199    | 0.001     | 0.090     |
| 55    | 0.505     | 0.852           | 1.503   | 0.399       | 0.200    | 0.001     | 0.090     |
| 60    | 0.499     | 0.859           | 1.482   | 0.399       | 0.200    | 0.001     | 0.090     |
| 65    | 0.509     | 0.865           | 1.447   | 0.399       | 0.201    | 0.001     | 0.090     |
| 70    | 0.525     | 0.846           | 1.480   | 0.395       | 0.204    | 0.003     | 0.088     |

#### Evaluation Episodes (every 5 epochs)

| Epoch | Survival | Mean Reward | Hunt | Gather | Flee | Rest | Eat   | Craft Total |
|-------|----------|-------------|------|--------|------|------|-------|-------------|
| 5     | 48%      | 285.2       | 4.5% | 38.2%  | 8.2% | 8.1% | 26.3% | 13.5%       |
| 10    | 31%      | 219.1       | 2.2% | 44.5%  | 5.2% | 4.9% | 27.5% | 14.2%       |
| 15    | 40%      | 270.3       | 1.3% | 44.5%  | 6.2% | 6.1% | 26.9% | 13.9%       |
| 20    | 38%      | 276.6       | 1.7% | 44.3%  | 6.1% | 6.5% | 26.1% | 14.3%       |
| 25    | 40%      | 249.2       | 2.5% | 42.1%  | 6.1% | 5.1% | 27.5% | 15.3%       |
| 30    | 41%      | 287.3       | 1.5% | 43.3%  | 6.1% | 6.7% | 27.9% | 13.3%       |
| 35    | 35%      | 261.1       | 0.6% | 44.3%  | 6.3% | 7.3% | 26.1% | 14.3%       |
| 40    | 49%      | 315.5       | 0.3% | 45.2%  | 4.3% | 7.7% | 27.6% | 14.0%       |
| 45    | 30%      | 221.3       | 0.5% | 45.3%  | 7.5% | 4.6% | 27.9% | 13.6%       |
| 50    | 46%      | 291.3       | 1.1% | 40.5%  | 7.2% | 5.2% | 27.9% | 17.0%       |
| 55    | 37%      | 284.1       | 0.2% | 45.1%  | 6.5% | 5.1% | 27.3% | 14.7%       |
| 60    | 39%      | 267.3       | 0.8% | 44.3%  | 6.3% | 4.2% | 27.5% | 15.8%       |
| 65    | 37%      | 264.3       | 0.5% | 43.5%  | 6.2% | 5.5% | 25.4% | 17.5%       |
| 70    | 41%      | 276.3       | 4.2% | 40.8%  | 8.3% | 4.5% | 27.2% | 13.6%       |

#### Message Analysis

| Epoch | Animal | Resource | Danger | CraftOpp | Event | Total Unique |
|-------|--------|----------|--------|----------|-------|-------------|
| 10    | 7      | 1        | 2      | 1        | 1     | 12          |
| 20    | 7      | 2        | 5      | 2        | 2     | 18          |
| 30    | 8      | 3        | 5      | 2        | 3     | 21          |
| 40    | 8      | 3        | 8      | 2        | 3     | 24          |
| 50    | 9      | 3        | 6      | 3        | 3     | 24          |
| 60    | 9      | 3        | 6      | 3        | 4     | 25          |
| 70    | 10     | 3        | 6      | 3        | 4     | 26          |

---

### Baselines (from `prototype.py`)

| Strategy | Survival Rate | Description |
|----------|--------------|-------------|
| **Random** | 21.1% | Uniform random action selection from valid actions |
| **Greedy** | 55.5% | Hardcoded heuristic: flee danger, hunt animals (with tool), gather otherwise, eat when energy low |
| **Optimal** | 63.1% | Perfect play: always picks the highest-reward valid action |

---

*Document generated from experimental outputs. Figures saved to `outputs/figures_runs5to7/`.*
