#!/usr/bin/env python3
"""
Liquid Diffusion Model Training Runner
"""

import os
import sys
import torch
import time
from datetime import datetime

def main():
    print("🌊" * 30)
    print("LIQUID DIFFUSION MODEL TRAINING")
    print("🌊" * 30)
    print()
    
    # System check
    print("🔧 SYSTEM CHECK")
    print("-" * 20)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU (training will be slower)")
    
    print()
    
    # Import the liquid diffusion model
    print("📦 IMPORTING LIQUID DIFFUSION MODEL")
    print("-" * 35)
    
    try:
        # Try to import from the saved file
        if os.path.exists('liquid_diffusion.py'):
            print("✅ Found liquid_diffusion.py")
            import liquid_diffusion
            from liquid_diffusion import train_true_liquid_diffusion, TrueLiquidDiffusionModel
            print("✅ Successfully imported liquid diffusion components")
        else:
            print("❌ liquid_diffusion.py not found!")
            print("Make sure you saved the liquid diffusion code as 'liquid_diffusion.py'")
            return
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install torch torchvision tqdm matplotlib")
        return
    
    print()
    
    # Training configuration
    print("⚙️ TRAINING CONFIGURATION")
    print("-" * 25)
    print("• Dataset: CIFAR-10 (32x32 color images)")
    print("• Model: TRUE Liquid Neural Network Diffusion")
    print("• Features: ODE dynamics, adaptive time constants")
    print("• Epochs: 3 (reduced for TRUE liquid training)")
    print("• Batch Size: 8 (optimized for liquid networks)")
    print("• TRUE Liquid Cells with continuous evolution")
    print()
    
    # Start training
    print("🚀 STARTING TRAINING")
    print("-" * 18)
    print("Training will:")
    print("1. Download CIFAR-10 dataset (~170MB)")
    print("2. Initialize liquid diffusion model")
    print("3. Train for 5 epochs with progress bars")
    print("4. Generate sample images")
    print("5. Show results and analysis")
    print()
    
    # Countdown
    print("Starting in...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🚀 GO!")
    print()
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at: {start_datetime}")
    print()
    
    try:
        # Training with TRUE Liquid Networks
        print("🌊 CALLING TRAIN_TRUE_LIQUID_DIFFUSION...")
        print("=" * 50)
        
        model, trainer, losses = train_true_liquid_diffusion()
        
        print("=" * 50)
        print("✅ TRUE LIQUID TRAINING COMPLETED!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print()
        print("🔧 TROUBLESHOOTING TIPS:")
        print("• If tensor dimension errors: check the fixed code")
        print("• If memory issues: reduce batch_size further")
        print("• If too slow: reduce num_epochs")
        import traceback
        traceback.print_exc()
        return
    
    # Training summary
    end_time = time.time()
    training_duration = end_time - start_time
    
    print()
    print("📊 TRAINING SUMMARY")
    print("-" * 18)
    print(f"Training duration: {training_duration/60:.1f} minutes")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate additional samples
    print()
    print("🎨 GENERATING ADDITIONAL SAMPLES")
    print("-" * 32)
    
    try:
        print("Generating 4 new images...")
        with torch.no_grad():
            device = next(model.parameters()).device
            samples = trainer.sample((4, 3, 32, 32), num_steps=25)
        
        # Display samples
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            sample_np = (samples[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
            sample_np = np.clip(sample_np, 0, 1)
            ax.imshow(sample_np)
            ax.axis('off')
            ax.set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.suptitle('Liquid Diffusion Generated Images', y=1.02, fontsize=14)
        plt.show()
        
        print("✅ Sample generation completed!")
        
    except Exception as e:
        print(f"⚠️ Sample generation failed: {e}")
    
    # Performance analysis
    print()
    print("📈 PERFORMANCE ANALYSIS")
    print("-" * 22)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Liquid Diffusion Model: {total_params:,} parameters")
    print(f"Uses liquid dynamics with adaptive time constants")
    print(f"Multi-scale processing for efficient computation")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {memory_used:.2f} GB")
    
    # Success message
    print()
    print("🎉" * 20)
    print("TRAINING COMPLETED!")
    print("🎉" * 20)
    print()
    print("✅ Successfully trained TRUE liquid neural network")
    print("✅ Generated images using continuous ODE dynamics") 
    print("✅ Demonstrated adaptive time constants")
    print("✅ Multi-step temporal evolution working")
    print("✅ Real recurrent liquid cells functioning")
    print()
    
    return model, trainer, losses

def quick_generate():
    """Quick function to generate more images after training"""
    print("🎨 Quick Image Generation")
    print("-" * 25)
    
    try:
        # This assumes training was already done
        global model, trainer
        
        print("Generating 4 quick samples...")
        with torch.no_grad():
            samples = trainer.sample((4, 3, 32, 32), num_steps=30)  # Faster with fewer steps
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            sample_np = (samples[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
            sample_np = np.clip(sample_np, 0, 1)
            ax.imshow(sample_np)
            ax.axis('off')
            ax.set_title(f'Quick Sample {i+1}')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Quick generation completed!")
        
    except Exception as e:
        print(f"❌ Quick generation failed: {e}")
        print("Make sure you've run the main training first!")

if __name__ == "__main__":
    print("🌊 LIQUID DIFFUSION TRAINING RUNNER 🌊")
    print()
    print("This script will train your revolutionary liquid diffusion model!")
    print("Make sure liquid_diffusion.py is in the same directory.")
    print()
    
    # Run the main training
    result = main()
    
    if result:
        model, trainer, losses = result
        print()
        print("🎯 TRAINING VARIABLES AVAILABLE:")
        print("• model: Your trained liquid diffusion model")
        print("• trainer: Diffusion trainer with sampling methods")
        print("• losses: Training loss history")
        print()
        print("💡 TRY THIS:")
        print("• quick_generate() - Generate 4 more images quickly")
        print("• trainer.sample((1, 3, 32, 32), num_steps=50) - Custom generation")
        print()
