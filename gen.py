#!/usr/bin/env python3
"""
Generate and save samples from trained ODE diffusion model
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from liquid_diffusion import TrueLiquidDiffusionModel, SimpleDiffusionTrainer

def load_and_generate():
    print("🎨 Generating ODE Diffusion Samples")
    print("=" * 35)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Recreate the model (same as training)
    model = TrueLiquidDiffusionModel(
        image_channels=3,
        model_channels=24,
        num_blocks=2,
        time_emb_dim=48
    ).to(device)
    
    # Create trainer
    trainer = SimpleDiffusionTrainer(model, device, num_timesteps=400)
    
    print("⚠️  Note: This creates a NEW untrained model")
    print("   To use your trained model, you'd need to save/load weights")
    print()
    
    # Generate samples with untrained model (will be noise-like)
    print("Generating 4 samples...")
    
    with torch.no_grad():
        samples = trainer.sample((4, 3, 32, 32), num_steps=25)
    
    # Save individual images
    for i in range(4):
        sample_np = (samples[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        sample_np = np.clip(sample_np, 0, 1)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_np)
        plt.axis('off')
        plt.title(f'ODE Diffusion Sample {i+1}')
        plt.savefig(f'ode_diffusion_sample_{i+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # Also save a grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        sample_np = (samples[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        sample_np = np.clip(sample_np, 0, 1)
        ax.imshow(sample_np)
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.suptitle('ODE-Inspired Diffusion Model Samples', y=1.02)
    plt.savefig('ode_diffusion_grid.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print("✅ Saved files:")
    print("   • ode_diffusion_sample_1.png")
    print("   • ode_diffusion_sample_2.png") 
    print("   • ode_diffusion_sample_3.png")
    print("   • ode_diffusion_sample_4.png")
    print("   • ode_diffusion_grid.png (2x2 grid)")
    print()
    print("📝 Note: These are from an untrained model (noise-like)")
    print("   To get good samples, you need to save/load trained weights")

if __name__ == "__main__":
    load_and_generate()
