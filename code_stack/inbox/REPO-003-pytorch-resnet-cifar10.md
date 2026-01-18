# Repo: PyTorch ResNet CIFAR10

- **ID**: REPO-003
- **GitHub**: https://github.com/akamaster/pytorch_resnet_cifar10
- **Paper**: He et al. 2015 - Deep Residual Learning for Image Recognition (arXiv:1512.03385)
- **Added**: 2026-01-18
- **Status**: reviewed

## Why Review

This repository provides a proper implementation of ResNet for CIFAR10 that matches the original He et al. 2015 paper specifications. Most PyTorch implementations adapt ImageNet ResNets which have different architectures and parameter counts. This is a canonical reference implementation for small-scale ResNet experiments.

**Lab Relevance**: ResNets on CIFAR10 are a standard testbed for our research on:
- Neural race dynamics (multi-view learning on small-scale architectures)
- Knowledge distillation baselines (teacher-student experiments)
- Feature learning and representation emergence studies

Having a correct reference implementation is critical for fair comparisons and baseline experiments.

## Key Methods to Understand
- [ ] CIFAR-specific architecture adaptations (Option A identity shortcuts with zero-padding)
- [ ] Training recipe (SGD + momentum, MultiStepLR, weight decay, data augmentation)
- [ ] Model variants (ResNet20/32/44/56/110/1202 with correct parameter counts)

## Initial Notes

### Repository Structure
- **resnet.py**: Model architecture definitions (BasicBlock, ResNet, resnet20/32/44/56/110/1202 factories)
- **trainer.py**: Training loop with standard CIFAR10 data loading and optimization
- **pretrained_models/**: Pre-trained checkpoints for all model variants
- **hubconf.py**: PyTorch Hub integration for easy model loading

### Key Implementation Details

**Architecture Differences from ImageNet ResNet**:
- Uses 3x3 conv with stride 1 at input (not 7x7 conv + maxpool)
- Starts with 16 filters (not 64)
- Uses "Option A" identity shortcuts: zero-padding instead of 1x1 projection convolutions
- Three stage architecture with [16, 32, 64] filters

**Training Setup**:
- 200 epochs with batch size 128
- SGD with momentum 0.9, weight decay 1e-4
- Learning rate: 0.1 initially, reduced at epochs 100 and 150 (MultiStepLR)
- Data augmentation: random horizontal flip, random crop with padding
- ResNet110/1202 use warmup (lr=0.01 for first epoch)

**Pretrained Models Performance** (test error %):
- ResNet20: 8.27% (paper: 8.75%)
- ResNet32: 7.37% (paper: 7.51%)
- ResNet44: 6.90% (paper: 7.17%)
- ResNet56: 6.61% (paper: 6.97%)
- ResNet110: 6.32% (paper: 6.43%)
- ResNet1202: 6.18% (paper: 7.93%)

All models match or exceed paper performance, confirming correct implementation.

### Code Quality Initial Assessment
- **Readability**: Excellent - clean, well-commented, follows PyTorch conventions
- **Documentation**: Adequate - good README, inline comments explain CIFAR-specific choices
- **Dependencies**: Minimal - just PyTorch, torchvision
- **Reproducibility**: Easy - clear training script, pretrained models, documented hyperparameters

### Size & Complexity
- 3 Python files (resnet.py ~159 lines, trainer.py ~306 lines, hubconf.py ~5 lines)
- Repository size: ~188MB (includes pretrained model weights)
- Complexity: Simple - straightforward training script, no complex abstractions

### Potential Reuse
- Baseline teacher models for knowledge distillation experiments
- Architecture reference for neural race multi-view experiments on CIFAR10
- Training recipe baseline for feature learning studies
- Pretrained checkpoints for initialization or probing experiments

### Notes on CIFAR-Specific Design
The repository emphasizes correctness of the CIFAR10-specific ResNet architecture:
- Option A shortcuts (zero-padding) reduce parameters vs Option B (projection)
- Smaller input conv (3x3) preserves spatial resolution for 32x32 images
- Confirms that most web implementations incorrectly use ImageNet architecture on CIFAR10

### Last Updated
2021-07-20 (no recent updates, but code is stable and widely used)
