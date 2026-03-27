# Command for long run
cd /cs/home/nc212/Documents/Fourth_Year/Dissertation

tmux new-session -d -s no_recon_run \
'cd /cs/home/nc212/Documents/Fourth_Year/Dissertation && \
systemd-inhibit --what=shutdown:sleep --why="long survival training" \
bash -lc "./run_experiments.sh 2>&1 | tee run_experiments_$(date +%Y%m%d_%H%M%S).out"'

cd /cs/home/nc212/Documents/Fourth_Year/Dissertation
tail -f "$(ls -t run_experiments_*.out | head -n 1)"
nvidia-smi

tmux kill-session -t no_recon_run

# Survival Game Debug Q&A Log


Comands ran:


CUDA_VISIBLE_DEVICES=0 python -m egg.zoo.survival_game.train --mode gs --sender_hidden 128 --receiver_hidden 128 --sender_embedding 32 --receiver_embedding 32 --sender_cell lstm --receiver_cell lstm --vocab_size 50 --max_len 2 --temperature 2.0 --temperature_decay 1.0 --temperature_minimum 0.1 --recon_weight 1.5 --action_entropy_coeff 0.1 --action_temperature 2.0 --reward_normalise --reward_scale 0.5 --lr 0.001 --batch_size 64 --n_epochs 50 --n_episodes 10000 --max_turns 20 --eval_freq 5 --track_topsim --topsim_max_samples 1000 --analyze_freq 10 --top_k_messages 10 --output_dir outputs --run_name run14 --log_file outputs/train_run14.log --message_progression_file outputs/message_progression_run14.jsonl --final_snapshot_file outputs/message_snapshot_final_run14.json --random_seed 14 --data_seed 14

CUDA_VISIBLE_DEVICES=0 python -m egg.zoo.survival_game.train --mode gs --sender_hidden 128 --receiver_hidden 128 --sender_embedding 32 --receiver_embedding 32 --sender_cell lstm --receiver_cell lstm --vocab_size 50 --max_len 2 --temperature 2.0 --temperature_decay 1.0 --temperature_minimum 0.1 --recon_weight 1.5 --action_entropy_coeff 0.1 --action_temperature 2.0 --reward_normalise --reward_scale 0.5 --lr 0.001 --batch_size 64 --n_epochs 50 --n_episodes 10000 --max_turns 20 --eval_freq 5 --track_topsim --topsim_max_samples 1000 --analyze_freq 10 --top_k_messages 10 --output_dir outputs --run_name run15 --log_file outputs/train_run15.log --message_progression_file outputs/message_progression_run15.jsonl --final_snapshot_file outputs/message_snapshot_final_run15.json --random_seed 15 --data_seed 15

CUDA_VISIBLE_DEVICES=0 python -m egg.zoo.survival_game.train --mode gs --sender_hidden 128 --receiver_hidden 128 --sender_embedding 32 --receiver_embedding 32 --sender_cell lstm --receiver_cell lstm --vocab_size 50 --max_len 2 --temperature 2.0 --temperature_decay 1.0 --temperature_minimum 0.1 --recon_weight 1.5 --action_entropy_coeff 0.1 --action_temperature 2.0 --reward_normalise --reward_scale 0.5 --lr 0.001 --batch_size 64 --n_epochs 50 --n_episodes 10000 --max_turns 20 --eval_freq 5 --track_topsim --topsim_max_samples 1000 --analyze_freq 10 --top_k_messages 10 --output_dir outputs --run_name run16 --log_file outputs/train_run16.log --message_progression_file outputs/message_progression_run16.jsonl --final_snapshot_file outputs/message_snapshot_final_run16.json --random_seed 16 --data_seed 16

CUDA_VISIBLE_DEVICES=0 python -m egg.zoo.survival_game.train --mode gs --sender_hidden 128 --receiver_hidden 128 --sender_embedding 32 --receiver_embedding 32 --sender_cell lstm --receiver_cell lstm --vocab_size 50 --max_len 2 --temperature 2.0 --temperature_decay 1.0 --temperature_minimum 0.1 --recon_weight 1.5 --action_entropy_coeff 0.1 --action_temperature 2.0 --reward_normalise --reward_scale 0.5 --lr 0.001 --batch_size 64 --n_epochs 50 --n_episodes 10000 --max_turns 20 --eval_freq 5 --track_topsim --topsim_max_samples 1000 --analyze_freq 10 --top_k_messages 10 --output_dir outputs --run_name run17 --log_file outputs/train_run17.log --message_progression_file outputs/message_progression_run17.jsonl --final_snapshot_file outputs/message_snapshot_final_run17.json --random_seed 17 --data_seed 17

CUDA_VISIBLE_DEVICES=0 python -m egg.zoo.survival_game.train --mode gs --sender_hidden 128 --receiver_hidden 128 --sender_embedding 32 --receiver_embedding 32 --sender_cell lstm --receiver_cell lstm --vocab_size 50 --max_len 2 --temperature 2.0 --temperature_decay 1.0 --temperature_minimum 0.1 --recon_weight 1.5 --action_entropy_coeff 0.1 --action_temperature 2.0 --reward_normalise --reward_scale 0.5 --lr 0.001 --batch_size 64 --n_epochs 50 --n_episodes 10000 --max_turns 20 --eval_freq 5 --track_topsim --topsim_max_samples 1000 --analyze_freq 10 --top_k_messages 10 --output_dir outputs --run_name run18 --log_file outputs/train_run18.log --message_progression_file outputs/message_progression_run18.jsonl --final_snapshot_file outputs/message_snapshot_final_run18.json --random_seed 18 --data_seed 18


Date: 2026-03-23

## archs.py

### Q1. Can we describe sender and receiver in simple English?
Answer:
- Sender is a neural network with two linear layers. The first linear layer is followed by a rectified linear activation function, and then another linear layer produces the sender hidden representation.
- Receiver is a neural network that combines message representation plus game state and then uses two linear layers (with rectified linear activation between them) to produce action scores.
- When reconstruction head is enabled, receiver has an extra prediction branch that predicts which exact entity was seen.

### Q2. Do we do backpropagation? I only see forward.
Answer:
- Yes. Forward methods define computation only.
- Parameter updates happen in training loop through EGG trainer in [train.py](train.py), where loss is computed and optimizer updates model weights.
- Standard pipeline is: forward pass -> loss -> backward pass -> optimizer step.

### Q3. If we mask illegal actions with very negative values, does that affect the rest of outputs?
Answer:
- It suppresses only illegal action logits by setting them to a very large negative value.
- After softmax, illegal actions get near-zero probability and valid actions absorb probability mass.
- This is intentional to enforce valid-action behavior.

### Q4. What is reconstruction head and why optional?
Answer:
- Reconstruction head is an auxiliary classifier that predicts entity identity from the message representation.
- It is optional because it is mainly used in Gumbel-Softmax mode to force messages to carry semantic content.
- If disabled, receiver only predicts actions.

## callbacks.py

### Q5. Are distractors used by sender in evaluation?
Answer:
- No. In [callbacks.py](callbacks.py), sender receives only target entity vector.
- Distractors are generated but not fed to sender or receiver policy in current implementation.

### Q6. Why EOS handling with max_len + 1 if max_len is 2?
Answer:
- In Gumbel-Softmax wrapper, message tensor includes an additional time step for end-of-sequence marker.
- So with max length 2 content tokens, tensor length can be 3 steps including termination handling.

### Q7. Comment says receiver output is 16 dimensions (11 action + 5 recon). Is that correct?
Answer:
- This comment is stale/inaccurate.
- Current receiver reconstruction head predicts 40 entities (not 5), so GS output with reconstruction is action logits + entity logits.
- Reference: [archs.py](archs.py).

### Q8. Is survival metric used in training loss?
Answer:
- No. Survival metric in evaluator is reporting-only.
- Training loss is defined in [losses.py](losses.py), based on reward/action terms and reconstruction terms.

### Q9. Why hard check messages.dim() == 3 in MessageAnalyzer?
Answer:
- Because GS messages are distributions over vocabulary with shape N x time x vocabulary (3 dimensions).
- Analyzer converts those soft/hard distributions to token ids using argmax for counting.
- In Reinforce, messages are already token ids, so shape differs.

## data.py

### Q10. Many imports from prototype are unused. Is this intentional?
Answer:
- Partly legacy convenience import style; several are unused in this file now.
- It does not break functionality but can be cleaned later for clarity.

### Q11. In encode_labels, what are entity vector and game state vector, and why these fields?
Answer:
- Entity vector is the six-value semantic description of encounter object (type, subtype, danger, energy class, tool requirement, weather dependency).
- Game state vector is normalized state features (energy, health, weather, turn progress, inventory values).
- Entity index is used for exact identity supervision and lookup.
- Weather, turn, max turns provide extra context needed to reconstruct state and compute reward logic consistently in loss.

### Q12. Are these labels for sender or receiver?
Answer:
- They are supervision payload for the loss function, not direct input features to sender/receiver neural modules.
- Sender input is one-hot entity vector; receiver input is state plus valid-action mask.

### Q13. Why simulate episodes in data generation file?
Answer:
- To create realistic state trajectories (inventory changes, weather changes, survival dynamics) before flattening into turn-level samples.

### Q14. Is loop logic duplicated with evaluator?
Answer:
- Similar structure exists in two places by design:
- Data simulation loop builds dataset with random policy transitions.
- Evaluator loop runs trained policy for reporting.

### Q15. Random action selection during simulation: is it used?
Answer:
- Yes. It advances state stochastically so generated dataset covers diverse situations.

### Q16. Difference between legacy dataset generation and pre-built samples?
Answer:
- Legacy mode: dataset class can self-generate episodes internally.
- Pre-built mode: [data.py](data.py) factory creates split-aware samples first, then passes sample lists to dataset.
- Pre-built samples are in memory lists, not stored as separate files by default.

### Q17. "Why do we make them all valid" in valid_mask creation?
Answer:
- It does not make all actions valid.
- It initializes zeros, then sets only truly valid action indices to one.
- Receiver then uses this mask to block illegal actions.

### Q18. Are we already building receiver input in data.py?
Answer:
- Yes. Dataset builds receiver input tensor (state + valid mask) so training batches are ready.
- Sender does not build receiver input.

## games.py

### Q19. What does "force sender communication" mean?
Answer:
- With reconstruction head enabled in GS mode, receiver must decode entity identity from message representation.
- This pressures sender to encode meaningful information in messages instead of relying only on easy shortcuts.

### Q20. What is custom loss and how does temperature affect model/loss?
Answer:
- Custom loss is SurvivalLoss from [losses.py](losses.py).
- Sender temperature (GS wrapper) controls how discrete or smooth message sampling is.
- Action temperature (inside SurvivalLoss) controls softness of action probability distribution used to compute expected reward.

### Q21. Why do we have a separate action temperature?
Answer:
- It is different from sender message temperature.
- Sender temperature affects communication channel sampling.
- Action temperature affects decision distribution over actions in differentiable expected-reward computation.

## train.py

### Q22. Why so many arguments?
Answer:
- This file is acting as an experiment runner, not only a model definition.
- The argument list exposes architecture, optimization, data generation, evaluation, logging, analysis, and checkpoint controls so multiple experiments can be reproduced from CLI without editing code.
- The actual training pipeline is still short in main: build data -> build game -> build optimizer -> register callbacks -> trainer.train.

### Q23. Do we build game then optimize with core optimizer even if we did not code optimizer logic?
Answer:
- Yes. In [train.py](train.py#L295), game parameters are passed to core.build_optimizer, which comes from EGG core utilities.
- This is expected: your project defines model/game/loss specifics, while EGG provides generic training infrastructure (optimizer construction, loops, callback hooks, etc.).

### Q24. Why callbacks after game/optimizer, and what is temperature update doing here?
Answer:
- Callbacks depend on objects that already exist (game sender for temperature, trainer state for logs/checkpoints), so they are registered after game/optimizer creation in [train.py](train.py#L298).
- TemperatureUpdater is not an external script; it is an EGG callback class used in-process each epoch, configured in [train.py](train.py#L305).
- It updates sender Gumbel temperature on epoch end using decay/minimum rules from [egg/core/callbacks.py](egg/core/callbacks.py#L159).

### Q25. What is a checkpoint, and is it per episode or per epoch?
Answer:
- Checkpoint is a saved training snapshot: model weights + optimizer state (+ scheduler state), so training can resume or models can be analyzed later.
- In this setup it is epoch-based, not episode-based.
- Frequency is controlled by checkpoint_freq and executed in callback on_epoch_end; see [egg/core/callbacks.py](egg/core/callbacks.py#L208).

### Q26. Where does validation happen? Is validation dataset actually used?
Answer:
- Validation is used by Trainer automatically when validation_data is provided.
- In [train.py](train.py#L374), validation_data=val_loader is passed to core.Trainer.
- In trainer loop, after each training epoch (subject to validation_freq), Trainer runs eval over validation_data and then calls on_validation_end callbacks; see [egg/core/trainers.py](egg/core/trainers.py#L278).
- Test set is separate and run manually after training in [train.py](train.py#L386).

## losses.py

### Q27. When does fallback entity reconstruction happen?
Answer:
- Fallback runs only when entity_idx from labels is out of bounds (for example -1), in _reconstruct_entity.
- It is not about sender/receiver prediction failure; it is purely a label integrity fallback path.
- With current data pipeline, entity_idx is usually valid because labels are built from ALL_ENTITIES.
- Important risk: fallback constructor currently passes unsupported keyword args to Entity (entity_type, subtype, etc.), while Entity dataclass in [prototype.py](prototype.py#L151) expects name/vector/description/inventory_key/inventory_qty. If fallback is triggered, it will likely raise TypeError.

### Q28. Why are spear/fire/shelter/rod decoded via > 0.5? Is that probability?
Answer:
- No, these are binary tool flags stored as normalized floats in state vector.
- During reconstruction, sv[4..7] are thresholded at 0.5 to recover booleans (has tool or not).
- It is a decode step from normalized tensor back to Inventory booleans, not a learned probability.

### Q29. Full logic of _compute_single_reward
Answer:
- Step 1: get valid actions from current entity/state.
- Step 2: if selected action is invalid, return -15 immediately.
- Step 3: if action resolver is missing, return -15.
- Step 4: run resolver for immediate action reward and state mutation.
- Step 5: if encounter not addressed by the chosen action, apply unaddressed-danger penalty.
- Step 6: add alive/health/energy bonuses (alive +10, energy>70 +5, health>70 +3).
- Step 7: apply death penalties if state crossed zero (energy <= 0: -50, health <= 0: -300).
- Final total is returned as scalar reward plus info dictionary.

### Q30. Why expected rewards in GS, and why different calculations from single-step reward?
Answer:
- GS needs differentiable signal wrt logits. Argmax + sampled stochastic resolver would be non-differentiable/high-variance.
- So GS computes deterministic reward estimates for all actions, then combines with soft action probabilities to get expected reward.
- This reduces gradient variance and keeps training stable compared with sampling one action outcome.
- It is different from _compute_single_reward by design: expected version removes random sampling and uses deterministic approximations for risk/reward terms.

### Q31. Do all actions get evaluated each sample in GS?
Answer:
- Yes. _compute_expected_rewards returns a vector of length N_ACTIONS for each sample.
- Then action_probs from softmax(logits / action_temperature) weight that vector, and expected_reward is the weighted sum.

### Q32. If probability is small, why include that action reward at all?
Answer:
- Because gradients in softmax-based objectives come from all coordinates, scaled by their probabilities.
- Very low-probability actions contribute very little, but not exactly zero, which is useful for exploration and for nudging near-tied actions.
- This is standard policy optimization under stochastic policies: optimize expected return E_p[R], not only greedy argmax action.
- If probabilities collapsed to one-hot too early, learning can get stuck.

### Q33. Is _gs_loss the default of this repository?
Answer:
- No. Default mode in [train.py](train.py#L65) is rf.
- _gs_loss is used only when running with --mode gs.

### Q34. Are _gs_loss comments up to date?
Answer:
- Mostly yes for core idea (softmax-weighted expected reward over all actions).
- Minor incompleteness: comments focus on game loss term, but full loss also adds reconstruction cross-entropy and optional action-entropy term.
- That is implemented in _gs_loss after expected_reward computation.

### Q35. Can reward_matrix be useful as a metric?
Answer:
- Yes, very useful for diagnostics.
- Examples: per-action expected reward curves, action-gap between best/second-best, fraction of states where model argmax matches reward argmax, calibration between action_probs and reward ranking.
- Current logs expose expected_reward and argmax-based reward, but not full reward_matrix by default.

### Q36. Full _gs_loss logic walkthrough
Answer:
- Split receiver output into action logits (first 11) and entity reconstruction logits (last 40).
- Unpack labels to reconstruct entity/state per sample.
- Build reward_matrix by computing deterministic expected reward for each action per sample.
- Compute action_probs = softmax(action_logits / action_temperature).
- Compute expected_reward = sum(action_probs * reward_matrix).
- Game loss is -expected_reward scaled by reward_scale.
- Compute action entropy and add entropy regularization term (encourage exploration).
- Compute reconstruction CE loss for 40-way entity target to force informative messages.
- Final loss = game_loss + recon_weight * recon_loss + entropy_loss.
- Return aux metrics (expected_reward, recon metrics, action rates, validity).

## prototype.py

### Q37. Do we use prototype.py at all?
Answer:
- Yes. It is not only a standalone script; it is also a shared mechanics module imported by training files.
- Imports from prototype appear in [archs.py](archs.py#L25), [data.py](data.py#L38), [callbacks.py](callbacks.py#L33), and [losses.py](losses.py#L37).

### Q38. Is prototype game logic the same as the one used in training?
Answer:
- Core mechanics/constants/entities/actions are shared from prototype module, so major game rules are aligned.
- However, training path adds extra abstractions and approximations:
- losses.py has deterministic expected-reward approximation for GS (_compute_expected_rewards), which is not identical to one-step stochastic resolver outcome.
- data.py simulates random-policy episodes for dataset generation; evaluator simulates trained-policy episodes for reporting.
- So: same core world rules, but not every computation path is identical because training needs differentiable objectives and dataset preprocessing.

