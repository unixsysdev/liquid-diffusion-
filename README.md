# ODE-Inspired Diffusion Model

## What This Is

A toy diffusion model that replaces the standard U-Net with recurrent, ODE-flavored blocks. Uses Euler-integrated dynamics with adaptive time constants for image denoising.

## Files

- `liquid_diffusion.py` - ODE-inspired diffusion implementation with recurrent cells
- `liquid_diffusion_inspired.py` - CNN-based approach (standard convolutions)
- `train.py` - Training script
- `data/` - CIFAR-10 dataset (auto-downloaded)

## Architecture Details

**RealLiquidCell**: Recurrent state updated via Euler step:
```
τ * dh/dt = -h + tanh(W_rec * h + W_in * x)
```
- Input-dependent time constants τ
- 3-7 integration steps per forward pass
- Recurrent connections for temporal dynamics

**SpatialLiquidProcessor**: 
- Global spatial pooling + time embedding
- Three liquid cells (different step counts)
- Broadcast back to spatial dimensions

## Limitations

- **Tiny capacity**: Global pooling destroys spatial detail
- **Basic ODE solver**: Simple Euler integration only
- **Not true LNN**: Missing sparse gating, proper solvers, biophysical inspiration
- **Toy quality**: Expect blurry 32x32 images, not production results

## Quick Start

```bash
python train.py
```

## Requirements

```bash
pip install torch torchvision tqdm matplotlib
```

## Results

- Generates 32x32 CIFAR-10 images
- ~38K parameters (vs 500M+ in standard diffusion)
- Fast training (~10 minutes on CPU)
- Educational demonstration of ODE dynamics in diffusion

## Technical Notes

This is **not** a faithful implementation of Liquid Neural Networks from the literature. It's a toy model that uses ODE-flavored recurrent dynamics in place of standard diffusion architectures.

**For better results**: Embed liquid cells in a shallow U-Net with proper spatial processing and skip connections.

**What it demonstrates**: Continuous-time dynamics can work in diffusion models, even with basic implementation.
