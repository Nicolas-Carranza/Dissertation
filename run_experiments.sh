#!/bin/bash

set -euo pipefail

# Focused no-recon tuning sweep.
# Each config is evaluated across the same seed set for fair comparison.

OUTPUT_DIR="outputs/no_recon_tuning"
GPU_ID="0"
SEEDS=(24 25 26 27 28)

# Config format: name|max_len|temperature|action_entropy_coeff|action_temperature
CONFIGS=(
	"cfgA_len2_t2p0_ae0p3_at2p0|2|2.0|0.3|2.0"
	"cfgB_len2_t2p0_ae0p5_at2p0|2|2.0|0.5|2.0"
	"cfgC_len3_t2p0_ae0p5_at2p0|3|2.0|0.5|2.0"
)

mkdir -p "$OUTPUT_DIR"

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r CFG_NAME MAX_LEN TEMPERATURE ACTION_ENT ACTION_TEMP <<< "$cfg"

	for SEED in "${SEEDS[@]}"; do
		RUN_NAME="${CFG_NAME}_seed${SEED}"
		LOG_FILE="$OUTPUT_DIR/train_${RUN_NAME}.log"
		PROG_FILE="$OUTPUT_DIR/message_progression_${RUN_NAME}.jsonl"
		SNAP_FILE="$OUTPUT_DIR/message_snapshot_final_${RUN_NAME}.json"

		echo "============================================================"
		echo "Running $RUN_NAME"
		echo "  max_len=$MAX_LEN temperature=$TEMPERATURE action_entropy_coeff=$ACTION_ENT action_temperature=$ACTION_TEMP"
		echo "============================================================"

		CUDA_VISIBLE_DEVICES="$GPU_ID" python -m egg.zoo.survival_game.train \
			--mode gs \
			--sender_hidden 128 \
			--receiver_hidden 128 \
			--sender_embedding 32 \
			--receiver_embedding 32 \
			--sender_cell lstm \
			--receiver_cell lstm \
			--vocab_size 50 \
			--max_len "$MAX_LEN" \
			--temperature "$TEMPERATURE" \
			--temperature_decay 1.0 \
			--temperature_minimum 0.1 \
			--recon_weight 0 \
			--action_entropy_coeff "$ACTION_ENT" \
			--action_temperature "$ACTION_TEMP" \
			--reward_normalise \
			--reward_scale 0.5 \
			--lr 0.001 \
			--batch_size 64 \
			--n_epochs 50 \
			--n_episodes 10000 \
			--max_turns 20 \
			--eval_freq 5 \
			--track_topsim \
			--topsim_max_samples 1000 \
			--analyze_freq 10 \
			--top_k_messages 10 \
			--output_dir "$OUTPUT_DIR" \
			--run_name "$RUN_NAME" \
			--log_file "$LOG_FILE" \
			--message_progression_file "$PROG_FILE" \
			--final_snapshot_file "$SNAP_FILE" \
			--random_seed "$SEED" \
			--data_seed "$SEED"
	done
done

echo "All no-recon tuning runs completed successfully."