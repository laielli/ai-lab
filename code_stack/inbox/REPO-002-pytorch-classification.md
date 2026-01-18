# Repo: PyTorch Classification

- **ID**: REPO-002
- **GitHub**: https://github.com/bearpaw/pytorch-classification
- **Paper**: N/A (implementation repository)
- **Added**: 2026-01-18
- **Status**: reviewed (see summaries/REPO-002-pytorch-classification.md)

## Why Review

This is a well-structured PyTorch codebase providing unified training infrastructure for image classification on CIFAR-10/100 and ImageNet. It implements multiple canonical architectures (ResNet, PreResNet, ResNeXt, WRN, DenseNet, VGG, AlexNet) with consistent training recipes and comprehensive logging utilities.

Relevant to lab research for:
1. **Baseline implementations** for feature learning experiments
2. **Training infrastructure** that could be adapted for mechanistic studies
3. **Multi-architecture comparison** framework useful for studying representation learning across different network designs
4. **Clean CIFAR experiments** as testbed for theory validation

## Key Methods to Understand

- [ ] Training loop structure (cifar.py, imagenet.py)
- [ ] Logger and visualization utilities (utils/logger.py)
- [ ] Architecture implementations (models/cifar/)
  - [ ] ResNet variants (resnet.py, preresnet.py)
  - [ ] ResNeXt (resnext.py)
  - [ ] DenseNet (densenet.py)
  - [ ] Wide ResNets (wrn.py)
- [ ] Data loading and augmentation pipelines
- [ ] Learning rate scheduling and optimization setup

## Initial Notes

**Structure:**
- Simple, flat directory structure (cifar.py, imagenet.py as main scripts)
- Separate model packages for CIFAR vs ImageNet (models/cifar/, models/imagenet/)
- Utility package for logging, evaluation, progress bars (utils/)

**Tech Stack:**
- PyTorch (older codebase, circa 2019)
- torchvision for data and some model backbones
- matplotlib for visualization

**Size:**
- ~2,157 lines of Python code across 18 files
- Lightweight and focused

**Quality Indicators:**
- Well-documented training recipes (TRAINING.md)
- Comprehensive README with benchmark results
- Includes logging and visualization utilities
- Pretrained models available (OneDrive link)
- Last updated: January 2019 (archived/stable, not actively maintained)

**Notable Features:**
- Unified command-line interface across all architectures
- Progress bar with rich training information
- Training curve visualization and logging
- Multi-GPU support
- Comprehensive hyperparameter configurations for reproducing paper results

**Potential Lab Use Cases:**
1. **Baseline training infrastructure** - Could adapt for controlled experiments comparing architectures
2. **Feature learning studies** - Multiple architecture variants to test theoretical predictions about representation emergence
3. **Knowledge distillation experiments** - Teacher-student pairs across different architectures
4. **Gradient dynamics analysis** - Training loop could be instrumented to study learning dynamics
5. **CIFAR testbed** - Low-compute setting for rapid theoretical validation

**Gaps/Limitations:**
- Older codebase (2019) - may need updates for modern PyTorch
- No tests or reproducibility guarantees beyond training recipes
- Minimal documentation beyond README and training commands
- No explicit feature extraction or representation analysis utilities (would need to add hooks)
