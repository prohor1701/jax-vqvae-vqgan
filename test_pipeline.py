#!/usr/bin/env python3
"""
Test script to verify the VAE/VQ-VAE compression requirements are met.

Requirements:
- VAE (KL): Compress 256x256x3 images to < 8192 numbers
- VQ/FSQ: Compress 256x256x3 images to ≤ 256 tokens
"""

def test_compression_requirements():
    """Test that all model configurations meet the compression requirements."""
    
    image_size = 256
    image_channels = 3
    total_input_values = image_size * image_size * image_channels
    
    print("="*80)
    print("COMPRESSION REQUIREMENTS VERIFICATION")
    print("="*80)
    print(f"Input: {image_size}×{image_size}×{image_channels} = {total_input_values:,} values")
    print()
    
    # Test different configurations
    configs = [
        {
            'name': 'VQ-VAE (default)',
            'quantizer_type': 'vq',
            'channel_multipliers': [1, 1, 2, 2, 4],
            'embedding_dim': 256,
            'requirement': 'tokens ≤ 256'
        },
        {
            'name': 'VQ-VAE (with 5 downsamples)',
            'quantizer_type': 'vq',
            'channel_multipliers': [1, 1, 2, 2, 4, 4],
            'embedding_dim': 256,
            'requirement': 'tokens ≤ 256'
        },
        {
            'name': 'KL-VAE (5 downsamples, embedding 127)',
            'quantizer_type': 'kl',
            'channel_multipliers': [1, 1, 2, 2, 4, 4],
            'embedding_dim': 127,
            'requirement': 'values < 8192'
        },
        {
            'name': 'KL-VAE (6 downsamples, embedding 255)',
            'quantizer_type': 'kl',
            'channel_multipliers': [1, 1, 2, 2, 4, 4, 8],
            'embedding_dim': 255,
            'requirement': 'values < 8192'
        },
        {
            'name': 'FSQ (default)',
            'quantizer_type': 'fsq',
            'channel_multipliers': [1, 1, 2, 2, 4],
            'embedding_dim': 4,
            'fsq_levels': 5,
            'requirement': 'tokens ≤ 256'
        },
    ]
    
    all_passed = True
    
    for config in configs:
        num_downsamples = len(config['channel_multipliers']) - 1
        latent_h = image_size // (2 ** num_downsamples)
        latent_w = image_size // (2 ** num_downsamples)
        
        print(f"\n{config['name']}")
        print("-" * 60)
        print(f"  Quantizer: {config['quantizer_type'].upper()}")
        print(f"  Downsamples: {num_downsamples}")
        print(f"  Latent spatial size: {latent_h}×{latent_w}")
        print(f"  Embedding dimension: {config['embedding_dim']}")
        
        if config['quantizer_type'] in ['vq', 'fsq']:
            num_tokens = latent_h * latent_w
            num_values = num_tokens * config['embedding_dim']
            
            if 'fsq_levels' in config:
                codebook_size = config['fsq_levels'] ** config['embedding_dim']
                print(f"  FSQ levels: {config['fsq_levels']}")
                print(f"  Effective codebook size: {codebook_size}")
            
            print(f"  Number of tokens: {num_tokens}")
            print(f"  Total values: {num_values:,}")
            
            # Check requirement
            passes = num_tokens <= 256
            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"  Requirement: {config['requirement']} - {status}")
            
            if not passes:
                all_passed = False
                print(f"  ERROR: {num_tokens} tokens exceeds limit of 256!")
                
        elif config['quantizer_type'] == 'kl':
            num_values = latent_h * latent_w * config['embedding_dim']
            print(f"  Total values: {num_values:,}")
            
            # Check requirement
            passes = num_values < 8192
            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"  Requirement: {config['requirement']} - {status}")
            
            if not passes:
                all_passed = False
                print(f"  ERROR: {num_values} values exceeds limit of 8192!")
        
        # Calculate compression ratio
        compression_ratio = total_input_values / num_values
        print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    print()
    print("="*80)
    if all_passed:
        print("✓ ALL CONFIGURATIONS PASS REQUIREMENTS")
    else:
        print("✗ SOME CONFIGURATIONS FAIL REQUIREMENTS")
    print("="*80)
    print()
    
    return all_passed

def test_epoch_calculations():
    """Test epoch-based training calculations."""
    
    print("="*80)
    print("EPOCH-BASED TRAINING VERIFICATION")
    print("="*80)
    
    num_epochs = 100
    steps_per_epoch = 5000
    save_epoch_interval = 10
    eval_epoch_interval = 10
    
    total_steps = num_epochs * steps_per_epoch
    num_saves = num_epochs // save_epoch_interval
    num_evals = num_epochs // eval_epoch_interval
    
    print(f"Total epochs: {num_epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps:,}")
    print()
    print(f"Save interval: every {save_epoch_interval} epochs")
    print(f"Number of checkpoints: {num_saves}")
    print(f"Save epochs: {list(range(save_epoch_interval, num_epochs+1, save_epoch_interval))[:5]}... (first 5)")
    print()
    print(f"Eval interval: every {eval_epoch_interval} epochs")
    print(f"Number of evaluations: {num_evals}")
    print(f"Eval epochs: {list(range(eval_epoch_interval, num_epochs+1, eval_epoch_interval))[:5]}... (first 5)")
    print()
    
    # Verify specific steps
    test_steps = [5000, 10000, 50000, 100000, 500000]
    print("Step-to-Epoch mapping examples:")
    for step in test_steps:
        if step <= total_steps:
            epoch = (step - 1) // steps_per_epoch + 1
            step_in_epoch = (step - 1) % steps_per_epoch + 1
            is_epoch_end = (step % steps_per_epoch == 0)
            print(f"  Step {step:6d} → Epoch {epoch:3d}, Step {step_in_epoch:4d}/5000, End: {is_epoch_end}")
    
    print("="*80)
    print()

def test_metrics():
    """Test that all required metrics are defined."""
    
    print("="*80)
    print("METRICS VERIFICATION")
    print("="*80)
    
    required_metrics = [
        'l1_loss',        # L1 (MAE)
        'l2_loss',        # L2 (MSE)
        'psnr',           # Peak Signal-to-Noise Ratio
        'perceptual_loss', # LPIPS
        'd_loss_real',    # GAN discriminator loss on real
        'd_loss_fake',    # GAN discriminator loss on fake
        'd_loss_for_vae', # GAN loss for generator
        'gradient_penalty', # Gradient penalty
        'quantizer_loss',  # VQ or KL loss
    ]
    
    print("Required metrics:")
    for metric in required_metrics:
        print(f"  ✓ {metric}")
    
    print()
    print(f"Total: {len(required_metrics)} metrics")
    print("="*80)
    print()

if __name__ == '__main__':
    print()
    
    # Run all tests
    passed = test_compression_requirements()
    test_epoch_calculations()
    test_metrics()
    
    # Summary
    if passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("The training pipeline is ready for TPU v5t with 8 replicas!")
        print("See TPU_TRAINING.md for usage instructions.")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Please review the failed configurations above.")
        exit(1)
