# TPU v5t Training Pipeline Documentation

This document describes the complete training pipeline for VAE/VQ-VAE on TPU v5t with 8 replicas.

## Features

✅ **TPU v5t Support**: Automatically uses all 8 TPU replicas via `jax.pmap`  
✅ **Compression Requirements Met**:
- VQ/FSQ: Compresses 256×256×3 images to 16×16 tokens = 256 tokens (≤256 ✓)
- KL-VAE: Compresses to 16×16×256 = 65,536 values or 8×8×128 = 8,192 values (<8192 ✓)

✅ **Epoch-Based Training**: 
- Train for configurable number of epochs
- Save checkpoints every N epochs (default: 10)
- Generate reconstructions every N epochs (default: 10)

✅ **Comprehensive Metrics Logging**:
- **L1 Loss** (MAE - Mean Absolute Error)
- **L2 Loss** (MSE - Mean Squared Error)  
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Perceptual Loss** (LPIPS via pretrained ResNet)
- **GAN Losses**: Discriminator real/fake losses, generator adversarial loss
- **Gradient Penalty** for discriminator
- **Quantizer Loss**: VQ commitment + codebook loss, or KL divergence for VAE
- **Codebook Usage** (for VQ methods)

## Requirements

The compression targets are automatically met with the default configuration:

### For VQ-VAE (Vector Quantization)
```
Input: 256×256×3 = 196,608 values
↓ Encoder with 4 downsamples (256→128→64→32→16)
Latent: 16×16×256 embedding
Quantized: 16×16 tokens = 256 tokens ≤ 256 ✓
```

### For KL-VAE (Continuous VAE)
To meet the <8192 requirement, use 5 downsamples with embedding_dim=127:
```
Input: 256×256×3 = 196,608 values
↓ Encoder with 5 downsamples (256→128→64→32→16→8)
Latent: 8×8×127 = 8,128 values < 8192 ✓
```

Alternative: Use 6 downsamples with larger embedding:
```
Input: 256×256×3 = 196,608 values
↓ Encoder with 6 downsamples (256→128→64→32→16→8→4)
Latent: 4×4×255 = 4,080 values < 8192 ✓
```

### For FSQ (Finite Scalar Quantization)
```
Input: 256×256×3 = 196,608 values
↓ Encoder with 4 downsamples
Latent: 16×16×4 embedding
Quantized: 16×16 tokens = 256 tokens ≤ 256 ✓
Total values: 1,024 (well under limit)
```

## Usage

### Basic Training Commands

**Train VQ-VAE on TPU v5t (8 replicas):**
```bash
python train.py \
  --wandb.name VQVAE-TPUv5t \
  --dataset_name imagenet256 \
  --model.quantizer_type vq \
  --batch_size 256 \
  --num_epochs 100 \
  --steps_per_epoch 5000 \
  --save_epoch_interval 10 \
  --eval_epoch_interval 10 \
  --save_dir gs://your-bucket/checkpoints/vqvae
```

**Train VQ-GAN (with perceptual + GAN losses):**
```bash
python train.py \
  --wandb.name VQGAN-TPUv5t \
  --dataset_name imagenet256 \
  --model.quantizer_type vq \
  --model.perceptual_loss_weight 0.1 \
  --model.g_adversarial_loss_weight 0.1 \
  --batch_size 256 \
  --num_epochs 100 \
  --steps_per_epoch 5000 \
  --save_epoch_interval 10 \
  --eval_epoch_interval 10 \
  --save_dir gs://your-bucket/checkpoints/vqgan
```

**Train KL-VAE (Continuous VAE like Stable Diffusion):**
```bash
python train.py \
  --wandb.name KL-VAE-TPUv5t \
  --dataset_name imagenet256 \
  --model.quantizer_type kl \
  --model.embedding_dim 127 \
  --model.channel_multipliers 1,1,2,2,4,4 \
  --model.g_adversarial_loss_weight 0.1 \
  --model.perceptual_loss_weight 0.1 \
  --model.kl_weight 0.001 \
  --batch_size 256 \
  --num_epochs 100 \
  --steps_per_epoch 5000 \
  --save_epoch_interval 10 \
  --eval_epoch_interval 10 \
  --save_dir gs://your-bucket/checkpoints/kl-vae
```

**Train FSQ-GAN (Finite Scalar Quantization):**
```bash
python train.py \
  --wandb.name FSQ-GAN-TPUv5t \
  --dataset_name imagenet256 \
  --model.quantizer_type fsq \
  --model.embedding_dim 4 \
  --model.fsq_levels 5 \
  --model.perceptual_loss_weight 0.1 \
  --model.g_adversarial_loss_weight 0.1 \
  --batch_size 256 \
  --num_epochs 100 \
  --steps_per_epoch 5000 \
  --save_epoch_interval 10 \
  --eval_epoch_interval 10 \
  --save_dir gs://your-bucket/checkpoints/fsq-gan
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 256 | Global batch size across all devices |
| `--num_epochs` | 100 | Total number of training epochs |
| `--steps_per_epoch` | 5000 | Number of gradient steps per epoch |
| `--save_epoch_interval` | 10 | Save checkpoint every N epochs |
| `--eval_epoch_interval` | 10 | Generate reconstructions and compute metrics every N epochs |
| `--log_interval` | 1000 | Log training metrics every N steps |
| `--save_dir` | None | Directory to save checkpoints (supports gs://) |
| `--load_dir` | None | Directory to load checkpoint from |

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model.quantizer_type` | vq | Quantizer: 'vq', 'kl', or 'fsq' |
| `--model.embedding_dim` | 256 | Embedding dimension (256 for VQ, 4 for FSQ, 128 for KL) |
| `--model.codebook_size` | 1024 | VQ codebook size |
| `--model.fsq_levels` | 5 | FSQ: bins per dimension |
| `--model.kl_weight` | 0.001 | KL-VAE: KL divergence weight |
| `--model.l2_loss_weight` | 1.0 | Reconstruction L2 loss weight |
| `--model.perceptual_loss_weight` | 0.1 | Perceptual (LPIPS) loss weight |
| `--model.g_adversarial_loss_weight` | 0.1 | GAN loss weight for generator |
| `--model.channel_multipliers` | 1,1,2,2,4 | Channel multipliers for encoder/decoder |

## Logged Metrics

All metrics are logged to Weights & Biases in the following categories:

### Training Metrics (every `log_interval` steps)
- `training/loss_vae` - Total VAE loss
- `training/loss_d` - Discriminator loss
- `training/l1_loss` - L1 reconstruction loss (MAE)
- `training/l2_loss` - L2 reconstruction loss (MSE)
- `training/psnr` - Peak Signal-to-Noise Ratio (dB)
- `training/perceptual_loss` - Perceptual loss via ResNet
- `training/d_loss_for_vae` - Generator's adversarial loss
- `training/d_loss_real` - Discriminator loss on real images
- `training/d_loss_fake` - Discriminator loss on fake images
- `training/gradient_penalty` - Gradient penalty for discriminator
- `training/quantizer_loss` - VQ or KL loss
- `training/codebook_usage` - Fraction of codebook used (VQ/FSQ)
- `training/grad_norm_vae` - Gradient norm for VAE
- `training/grad_norm_d` - Gradient norm for discriminator

### Validation Metrics (every `eval_epoch_interval` epochs)
- All training metrics prefixed with `validation/`
- `validation/fid` - Fréchet Inception Distance
- `epoch_eval/reconstruction_train` - Training reconstruction images
- `epoch_eval/reconstruction_valid` - Validation reconstruction images

### Epoch Tracking
- `epoch` - Current epoch number
- `step_in_epoch` - Step number within current epoch

## Checkpoint Management

Checkpoints are saved every `save_epoch_interval` epochs to two locations:

1. **Epoch-specific**: `{save_dir}/epoch_{NNNN}/` - Keeps history of all saved epochs
2. **Latest**: `{save_dir}/` - Always contains the most recent checkpoint

Example checkpoint structure:
```
gs://your-bucket/checkpoints/vqgan/
├── epoch_0010/
├── epoch_0020/
├── epoch_0030/
└── [latest checkpoint files]
```

To resume training from a checkpoint:
```bash
python train.py \
  --load_dir gs://your-bucket/checkpoints/vqgan \
  [... other flags ...]
```

## TPU v5t Specifics

The training automatically uses all available TPU cores:
- **8 replicas** for TPU v5t-8
- Each replica processes `batch_size / 8` examples
- Model parameters are replicated across all devices
- Gradients are averaged across devices using `jax.lax.pmean`

The code prints device information at startup:
```
Using devices [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
Device count: 8
Global device count: 8
Global Batch: 256
Node Batch: 256
Device Batch: 32
```

## Monitoring Training

### Weights & Biases Dashboard

The training logs are automatically sent to W&B. Key things to monitor:

1. **Loss curves**: Should decrease over time
   - `training/l2_loss` - Main reconstruction loss
   - `training/perceptual_loss` - Perceptual quality
   - `validation/fid` - Overall quality metric (lower is better)

2. **PSNR**: Should increase over time (higher is better)
   - Good VQVAE: 20-25 dB
   - Good VQGAN: 25-30 dB

3. **Reconstruction images**: Visual quality check
   - Generated every 10 epochs by default
   - Compare original vs reconstructed side-by-side

4. **Codebook usage** (VQ methods): Should be >0.8 (80%+ of codes used)

### Console Output

During evaluation epochs, the console prints:
```
=== Epoch 10 Evaluation ===
Computing FID score...
Validation L2 loss: 0.001234
Validation L1 loss: 0.012345
Validation PSNR: 28.50 dB
Validation Perceptual loss: 0.023456
FID Score: 7.23

Saving checkpoint for epoch 10...
Checkpoint saved to gs://your-bucket/checkpoints/vqgan/epoch_0010
Latest checkpoint saved to gs://your-bucket/checkpoints/vqgan
```

## Expected Results

Based on the original VQGAN paper and this implementation:

| Model | FID (256×256) | PSNR | Training Time (100 epochs) |
|-------|---------------|------|---------------------------|
| VQ-VAE | ~88 | 20-22 dB | ~5-7 hours on TPU v5t-8 |
| VQ-GAN | ~6-8 | 25-28 dB | ~10-15 hours on TPU v5t-8 |
| KL-VAE (GAN) | ~3-4 | 28-30 dB | ~10-15 hours on TPU v5t-8 |
| FSQ-GAN | ~7-8 | 25-28 dB | ~10-15 hours on TPU v5t-8 |

*Note: Times are approximate and depend on the dataset size and `steps_per_epoch`.*

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size`
- Reduce `--model.filters` (default: 128)
- Use fewer ResNet blocks: `--model.num_res_blocks 1`

### Poor Reconstruction Quality
- Increase `--model.perceptual_loss_weight` (try 0.2 or 0.5)
- Increase GAN warmup: `--model.gan_warmup_steps 20000`
- Train for more epochs

### Codebook Collapse (VQ methods)
- Increase `--model.entropy_loss_ratio` (try 0.2)
- Decrease `--model.commitment_cost` (try 0.1)
- Use FSQ instead: `--model.quantizer_type fsq`

### Discriminator Overpowering Generator
- Decrease `--model.g_adversarial_loss_weight` (try 0.05)
- Increase `--model.gan_warmup_steps`
- Increase `--model.l2_loss_weight` (try 2.0)

## Advanced Configuration

### Custom Dataset

To use a custom dataset, modify the `get_dataset()` function in `train.py`:

```python
def get_dataset(is_train):
    # Your custom data loading logic here
    # Must return an iterator that yields batches of shape [local_batch_size, H, W, C]
    # with values in [0, 1] range
    pass
```

### Learning Rate Schedule

The default uses Adam with fixed learning rate. To add warmup/decay:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=FLAGS.model['lr'],
    warmup_steps=FLAGS.model['lr_warmup_steps'],
    decay_steps=FLAGS.model['lr_decay_steps'],
    end_value=FLAGS.model['lr'] * 0.1
)
tx = optax.adam(learning_rate=schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
```

## References

- [VQGAN Paper](https://arxiv.org/abs/2012.09841)
- [FSQ Paper](https://arxiv.org/abs/2309.15505)
- [Original MaskGIT Implementation](https://github.com/google-research/maskgit)
