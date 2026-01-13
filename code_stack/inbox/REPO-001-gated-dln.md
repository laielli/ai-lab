# Repo: Gated Deep Linear Networks

- **ID**: REPO-001
- **GitHub**: https://github.com/facebookresearch/gated-dln
- **Paper**: PAPER-001 (Saxe et al. 2022 - Neural Race Reduction)
- **Added**: 2026-01-13
- **Status**: reviewed
- **Summary**: code_stack/summaries/REPO-001-gated-dln.md

## Why Review

This is the official implementation of the Neural Race Reduction paper (Saxe et al., ICML 2022), which provides the theoretical foundation for our neural-race-multiview project. Understanding the implementation is critical for:

1. **Validating theoretical predictions** - seeing exactly how the authors implemented MSE + gradient flow dynamics
2. **Understanding implementation choices** - how they create competing pathways and measure pathway strength
3. **Identifying reusable components** - pathway tracking, multi-view data generation, visualization tools
4. **Clarifying theory-experiment gaps** - comparing their experimental setup to our classification experiments

The paper summary (PAPER-001) identified key mismatches between Saxe's theory (MSE loss, gradient flow) and standard training (cross-entropy, SGD). This codebase will reveal how the authors bridged theory to experiments.

## Key Methods to Understand

- [x] **Neural Race dynamics implementation** - CRITICAL FINDING: Experiments use CrossEntropyLoss, theory uses MSE
- [x] **GDLN framework** - Implemented with shared hidden layer, separate encoders/decoders
- [x] **Multi-view data generation** - Uses rotation and permutation transforms (NOT orthogonal)
- [x] **Gradient flow simulation** - Uses discrete SGD with momentum (NOT gradient flow)
- [x] **MSE loss implementation** - ONLY in notebooks; main code uses CrossEntropyLoss
- [x] **Pathway competition measurement** - Binary selected/unselected only, no SVD tracking
- [x] **Visualization tools** - Notebooks have RSA and pathway evolution plots

## Initial Notes

### Repository Metadata
- **Languages**: Jupyter Notebook (88.7%), Python (10.9%), Shell (0.4%)
- **License**: CC BY-NC 4.0 (non-commercial)
- **Status**: Archived (October 31, 2023) - no longer maintained
- **Stars**: 8 (low visibility, but official implementation)
- **Paper**: ICML 2022, Vol. 162, pp. 19287-19309

### Structure Overview
```
├── config/          # Experiment configurations
├── notebooks/       # Analysis notebooks (88.7% of repo!)
├── scripts/         # Experiment execution scripts
├── src/             # Core implementation
├── tests/           # Test suite
├── main.py          # Main training entry point
├── extract_features.py
├── resume.py
└── requirements.txt
```

### Tech Stack
- Python 3.9
- PyTorch 1.11.0
- Torchvision 0.12.0
- CUDA 11.3

### Experiments Mentioned
- MNIST experiments
- CIFAR-10 experiments
- Execution via shell scripts

### Key Questions for Review

1. **Loss function confirmation**: Do they actually use MSE or do experiments use cross-entropy?
2. **Gradient flow vs SGD**: How do they implement gradient flow? Small learning rate approximation?
3. **Saturation mechanism**: Is there explicit $s_{max}$ constraint in the code or does it emerge?
4. **Pathway tracking**: How do they measure pathway strengths during training?
5. **Gating pattern handling**: How do they ensure fixed gating patterns (assumption A3)?
6. **Data generation**: How do they create multi-view datasets with orthogonal structure?
7. **Reproducibility**: Are experiments easily reproducible with provided configs?

### Red Flags
- Archived repository suggests no ongoing maintenance
- Heavy reliance on notebooks (88.7%) may indicate less structured codebase
- Low star count may indicate limited community validation

### Relevance to neural-race-multiview
- **High priority review** - this is the theoretical foundation for our project
- Need to understand exactly where their experiments match theory and where they don't
- May provide pathway tracking code we can reuse
- Could inform our experimental design to better test race dynamics
