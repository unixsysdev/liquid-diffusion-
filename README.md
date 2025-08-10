# Liquid Neural Network Diffusion

## What This Is

Implementation of diffusion models using liquid neural networks (LNNs) for image generation. Tests whether continuous neural dynamics can replace traditional U-Net architectures in diffusion models.

## Files

- `liquid_diffusion.py` - **TRUE** liquid neural network implementation with ODE dynamics
- `liquid_diffusion_inspired.py` - CNN-based approach mimicking liquid networks (fake liquid)
- `train.py` - Training script
- `data/` - CIFAR-10 dataset (auto-downloaded)

## Key Features (True LNN)

- Continuous ODE dynamics: `τ * dh/dt = -h + activation(...)`
- Adaptive time constants based on input complexity
- Multi-step temporal evolution (3-7 Euler integration steps)
- Recurrent connections with temporal memory
- 38K parameters vs 500M+ in traditional diffusion models

## Quick Start

```bash
python train.py
```

## Requirements

```bash
pip install torch torchvision tqdm matplotlib
```

## Architecture

**Traditional Diffusion**: U-Net with discrete operations  
**This Implementation**: Liquid cells with continuous dynamics

```
Input → SpatialLiquidProcessor → TrueLiquidDiffusionModel → Denoised Output
         ↓
    RealLiquidCell (ODE dynamics)
    - Global liquid (slow, 3 steps)
    - Local liquid (medium, 5 steps)  
    - Detail liquid (fast, 7 steps)
```

## Results

- Generates 32x32 CIFAR-10 images
- ~100x parameter efficiency vs standard diffusion
- Uses biological-inspired continuous neural dynamics
- Proof of concept for liquid network image generation

## Training Details

- Dataset: CIFAR-10 (32x32 RGB)
- Batch size: 8
- Epochs: 3
- Timesteps: 400
- Device: CPU/CUDA
- Training time: ~10-30 minutes

## Technical Notes

This appears to be one of the first implementations combining liquid neural networks with diffusion models for image generation. The approach replaces traditional U-Net denoising with continuous ODE-based neural dynamics.

**Real LNN**: `liquid_diffusion.py` - Uses actual continuous dynamics  
**Fake LNN**: `liquid_diffusion_inspired.py` - CNN mimicking liquid behavior
