#!/bin/bash
# Example training script for TPU v5t with 8 replicas
# This script trains a VQ-GAN model on ImageNet 256x256

# Configuration
WANDB_PROJECT="vqgan-tpu"
EXPERIMENT_NAME="vqgan-tpu-v5t-run1"
SAVE_DIR="gs://your-bucket/checkpoints/vqgan-${EXPERIMENT_NAME}"

# Training parameters
BATCH_SIZE=256          # Global batch size (32 per TPU core with 8 cores)
NUM_EPOCHS=100          # Total epochs
STEPS_PER_EPOCH=5000    # Steps per epoch (~1.28M images with batch 256)
SAVE_EVERY=10           # Save checkpoint every N epochs
EVAL_EVERY=10           # Evaluate and generate reconstructions every N epochs

# Model configuration
QUANTIZER_TYPE="vq"     # Options: vq, kl, fsq
EMBEDDING_DIM=256       # 256 for VQ, 4 for FSQ, 128 for KL
CODEBOOK_SIZE=1024      # VQ codebook size

# Loss weights
L2_WEIGHT=1.0
PERCEPTUAL_WEIGHT=0.1
GAN_WEIGHT=0.1

echo "==============================================="
echo "Starting VQ-GAN Training on TPU v5t"
echo "==============================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Save directory: ${SAVE_DIR}"
echo "Total epochs: ${NUM_EPOCHS}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Total steps: $((NUM_EPOCHS * STEPS_PER_EPOCH))"
echo "Batch size: ${BATCH_SIZE}"
echo "==============================================="

# Run training
python train.py \
  --wandb.project "${WANDB_PROJECT}" \
  --wandb.name "${EXPERIMENT_NAME}" \
  --dataset_name imagenet256 \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --steps_per_epoch ${STEPS_PER_EPOCH} \
  --save_epoch_interval ${SAVE_EVERY} \
  --eval_epoch_interval ${EVAL_EVERY} \
  --save_dir "${SAVE_DIR}" \
  --model.quantizer_type ${QUANTIZER_TYPE} \
  --model.embedding_dim ${EMBEDDING_DIM} \
  --model.codebook_size ${CODEBOOK_SIZE} \
  --model.l2_loss_weight ${L2_WEIGHT} \
  --model.perceptual_loss_weight ${PERCEPTUAL_WEIGHT} \
  --model.g_adversarial_loss_weight ${GAN_WEIGHT} \
  --log_interval 100 \
  "$@"  # Pass any additional arguments

echo "==============================================="
echo "Training completed!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "==============================================="
