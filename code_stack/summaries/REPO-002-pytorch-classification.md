# Summary: PyTorch Classification

- **Repo ID**: REPO-002
- **GitHub**: https://github.com/bearpaw/pytorch-classification
- **Paper**: N/A (implementation repository)
- **Reviewed**: 2026-01-18

## One-Line Summary

A clean, well-organized PyTorch training infrastructure for CIFAR-10/100 and ImageNet classification with implementations of ResNet, Pre-activation ResNet, Wide ResNet (WRN), ResNeXt, DenseNet, VGG, and AlexNet.

## Codebase Overview

### Structure

```
pytorch-classification/
├── cifar.py              # Main CIFAR training script (353 lines)
├── imagenet.py           # Main ImageNet training script (similar structure)
├── models/
│   ├── cifar/            # CIFAR-adapted architectures (948 lines total)
│   │   ├── resnet.py     # ResNet for CIFAR (166 lines)
│   │   ├── preresnet.py  # Pre-activation ResNet (164 lines)
│   │   ├── wrn.py        # Wide ResNet (93 lines)
│   │   ├── resnext.py    # ResNeXt (125 lines)
│   │   ├── densenet.py   # DenseNet-BC (148 lines)
│   │   ├── vgg.py        # VGG variants (138 lines)
│   │   └── alexnet.py    # AlexNet (44 lines)
│   └── imagenet/         # ImageNet architectures (ResNeXt only)
└── utils/
    ├── logger.py         # Training curve logging and visualization
    ├── misc.py           # AverageMeter, mkdir, initialization
    ├── eval.py           # accuracy() function
    └── visualize.py      # Plotting utilities
```

### Tech Stack
- **Language**: Python 2.7/3.x compatible (uses `__future__` imports)
- **Framework**: PyTorch (circa 2017-2019, requires minor updates for modern PyTorch)
- **Key dependencies**: torch, torchvision, matplotlib, numpy
- **Progress bar**: Uses git submodule for `progress` library

### Size & Complexity
- **Lines of code**: ~2,157 Python lines
- **Core modules**: 7 model architectures + training infrastructure
- **Complexity assessment**: **Simple** - clean, readable code with consistent patterns

## Quality Assessment

### Code Quality
- **Readability**: **Good** - consistent naming, clear structure, but minimal docstrings
- **Documentation**: **Minimal** - README covers usage, TRAINING.md has recipes, but code lacks docstrings
- **Tests**: **None** - no test files
- **Reproducibility**: **Easy** - explicit training recipes with hyperparameters, random seed support

### Utility Assessment
- **Reusable components**: ResNet, WRN, training loop, logging utilities
- **Adaptation difficulty**: **Trivial to Moderate** - clean code but needs minor PyTorch API updates
- **Active maintenance**: **Archived** - last updated January 2019

### PyTorch Compatibility Notes

The code uses deprecated PyTorch APIs that need updating:
- `loss.data[0]` → `loss.item()`
- `torch.autograd.Variable` → no longer needed (automatic)
- `volatile=True` → `with torch.no_grad():`
- `cuda(async=True)` → `cuda(non_blocking=True)`

---

## Key Implementation Details

### ResNet (CIFAR variant)

**Location**: `models/cifar/resnet.py:L22-167`

**Architecture**: Post-activation ResNet following He et al. 2015, adapted for 32x32 CIFAR images.

**Key Structure**:
```python
# ResNet for CIFAR uses 3 stages with doubling width
# Total depth = 6n + 2 (for BasicBlock) or 9n + 2 (for Bottleneck)
# Example: ResNet-110 has n=18 blocks per stage

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        # Initial conv: 3x3, 16 channels (no maxpool - CIFAR is small!)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three stages: 16→16, 16→32 (stride 2), 32→64 (stride 2)
        self.layer1 = self._make_layer(block, 16, n)          # 32x32
        self.layer2 = self._make_layer(block, 32, n, stride=2) # 16x16
        self.layer3 = self._make_layer(block, 64, n, stride=2) # 8x8

        # Global average pooling (8x8→1x1) then FC
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
```

**BasicBlock (post-activation)**:
```python
# Conv → BN → ReLU → Conv → BN → (+residual) → ReLU
def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        residual = self.downsample(x)
    out += residual
    out = self.relu(out)  # ReLU AFTER addition
    return out
```

**Available depths**: 20, 32, 44, 56, 110, 1202 (BasicBlock); 20, 29, 47, 56, 110 (Bottleneck)

**Initialization**: Kaiming normal (He init) for conv weights, ones for BN gamma, zeros for BN beta.

---

### Pre-activation ResNet (PreResNet)

**Location**: `models/cifar/preresnet.py:L22-165`

**Difference from standard ResNet**: BN and ReLU come BEFORE convolutions (He et al. 2016).

```python
# Pre-activation order: BN → ReLU → Conv → BN → ReLU → Conv → (+residual)
def forward(self, x):
    residual = x
    out = self.bn1(x)    # BN first
    out = self.relu(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    if self.downsample is not None:
        residual = self.downsample(x)
    out += residual      # No ReLU after addition
    return out
```

**Key Difference**: Final BN+ReLU after all blocks (before classifier), not inside blocks.

---

### Wide ResNet (WRN)

**Location**: `models/cifar/wrn.py:L1-94`

**Architecture**: Pre-activation ResNet with width multiplier k (Zagoruyko & Komodakis 2016).

**Key Structure**:
```python
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        # Depth must satisfy: (depth - 4) % 6 == 0
        # WRN-28-10 means depth=28, widen_factor=10
        n = (depth - 4) // 6  # blocks per stage

        # Channel widths scaled by widen_factor
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        # WRN-28-10: [16, 160, 320, 640]

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])  # Final BN
```

**WRN BasicBlock (pre-activation with dropout)**:
```python
def forward(self, x):
    if not self.equalInOut:
        x = self.relu1(self.bn1(x))  # Pre-activation
    else:
        out = self.relu1(self.bn1(x))
    out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
    if self.droprate > 0:
        out = F.dropout(out, p=self.droprate, training=self.training)
    out = self.conv2(out)
    return torch.add(x if self.equalInOut else self.convShortcut(x), out)
```

**Configuration formulas**:
- WRN-d-k: depth=d, widen_factor=k
- WRN-28-10: 36.48M params, 3.79% error on CIFAR-10
- Blocks per stage: (depth - 4) / 6

**Shortcut handling**: Uses 1x1 conv shortcut only when dimensions change (efficient).

---

### Training Infrastructure

**Location**: `cifar.py:L1-353`

**Data Augmentation** (CIFAR standard):
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # Pad 4, then random 32x32 crop
    transforms.RandomHorizontalFlip(),        # 50% horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                        (0.2023, 0.1994, 0.2010)),  # CIFAR-10 std
])
```

**Training Loop**:
```python
# Standard SGD with momentum, step LR decay
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:  # e.g., [81, 122] for ResNet
        state['lr'] *= args.gamma  # e.g., 0.1
```

**Default Hyperparameters**:
| Architecture | Epochs | LR Schedule | Gamma | Weight Decay |
|--------------|--------|-------------|-------|--------------|
| ResNet-110 | 164 | [81, 122] | 0.1 | 1e-4 |
| WRN-28-10 | 200 | [60, 120, 160] | 0.2 | 5e-4 |
| DenseNet | 300 | [150, 225] | 0.1 | 1e-4 |

**Model instantiation** (architecture-specific):
```python
# WRN uses depth and widen_factor
if args.arch.startswith('wrn'):
    model = models.__dict__[args.arch](
        num_classes=num_classes,
        depth=args.depth,
        widen_factor=args.widen_factor,
        dropRate=args.drop,
    )
# ResNet uses depth and block_name
elif args.arch.endswith('resnet'):
    model = models.__dict__[args.arch](
        num_classes=num_classes,
        depth=args.depth,
        block_name=args.block_name,
    )
```

---

## Architecture Notes

### Data Flow (WRN-28-10 example)

```
Input (3, 32, 32)
    │
    ▼ conv1: 3→16 channels
Block1 (n=4 blocks): 16→160 channels, 32x32→32x32
    │
    ▼ stride=2
Block2 (n=4 blocks): 160→320 channels, 32x32→16x16
    │
    ▼ stride=2
Block3 (n=4 blocks): 320→640 channels, 16x16→8x8
    │
    ▼ BN → ReLU → AvgPool(8)
FC: 640→num_classes
```

### Key Abstractions

1. **Block classes**: `BasicBlock`, `Bottleneck`, `NetworkBlock` - encapsulate residual connection patterns
2. **`_make_layer()` method**: Factory for creating sequential blocks with proper downsampling
3. **Logger class**: TSV-based logging with matplotlib visualization
4. **AverageMeter**: Running statistics for loss and accuracy tracking

### Design Decisions

1. **No maxpool**: CIFAR images are 32x32, so no initial maxpool (unlike ImageNet variants)
2. **3 stages**: Fixed at 3 stages with spatial dimensions 32→16→8
3. **BN bias=False**: Conv layers have bias=False when followed by BatchNorm
4. **Kaiming init**: All models use He initialization for conv weights

---

## Relevance to Lab Research

### Potential Reuse

- [x] **WRN implementation for neural-race-multiview experiments** - Can extend current synthetic experiments to real CIFAR with WRN architecture to validate multi-view theory predictions
- [x] **Training infrastructure for feature learning studies** - Clean loop with logging could be instrumented for pathway analysis
- [x] **Baseline implementations** - Multiple architectures for controlled comparisons

### Integration Points for Neural Race Experiments

**Adding gradient/pathway hooks**:
```python
# Where to add hooks in WRN (models/cifar/wrn.py)

class WideResNet(nn.Module):
    def __init__(self, ...):
        ...
        # Add hooks storage
        self.activations = {}
        self.gradients = {}

    def forward(self, x):
        # Hook after each block to measure pathway strengths
        out = self.conv1(x)
        self.activations['conv1'] = out

        out = self.block1(out)
        self.activations['block1'] = out  # Pathway measurement point

        out = self.block2(out)
        self.activations['block2'] = out  # Pathway measurement point

        out = self.block3(out)
        self.activations['block3'] = out  # Pathway measurement point
        ...
```

**Adding knowledge distillation**:
```python
# In cifar.py, modify training loop:

def train_kd(trainloader, student, teacher, criterion_kd, optimizer, epoch, temperature=4.0):
    student.train()
    teacher.eval()

    for inputs, targets in trainloader:
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)

        # KD loss: KL(softmax(student/T) || softmax(teacher/T)) * T^2
        loss = criterion_kd(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Learnings

1. **Pre-activation vs post-activation matters**: PreResNet achieves significantly better results (6.11% vs 4.94% on CIFAR-10) - relevant for understanding gradient flow in neural race dynamics
2. **Width scaling is effective**: WRN-28-10 (36.48M params) outperforms ResNet-110 (1.70M params) substantially - width enables richer feature representations
3. **Dropout placement in WRN**: Between conv layers in blocks, not after pooling - different regularization dynamics

### Gaps or Limitations

1. **No built-in feature extraction**: Would need to add hooks for intermediate representations
2. **No knowledge distillation**: Would need to implement KD loss and teacher loading
3. **Dated PyTorch API**: Needs minor updates for PyTorch 2.x compatibility
4. **Single-view data only**: No multi-view data structure - would need to adapt data pipeline for multi-view experiments

---

## Cross-References

- **Paper Summary**: N/A (implementation repository, not tied to a specific paper)
- **Related Papers**:
  - He et al. 2015 (ResNet)
  - He et al. 2016 (Pre-activation ResNet)
  - Zagoruyko & Komodakis 2016 (Wide ResNet)
- **Lab Papers Using**: papers/neural-race-multiview/ (potential - WRN for real-data experiments)
- **Related Repos**: REPO-001-gated-dln (neural race theory implementation)

---

## Benchmark Results (from README)

| Model | Params (M) | CIFAR-10 (%) | CIFAR-100 (%) |
|-------|------------|--------------|---------------|
| ResNet-110 | 1.70 | 6.11 | 28.86 |
| PreResNet-110 | 1.70 | 4.94 | 23.65 |
| **WRN-28-10 (drop 0.3)** | **36.48** | **3.79** | **18.14** |
| ResNeXt-29, 8x64 | 34.43 | 3.69 | 17.38 |
| DenseNet-BC (L=100, k=12) | 0.77 | 4.54 | 22.88 |

---

## Quick Start Commands

```bash
# ResNet-110 on CIFAR-10
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 \
    --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110

# WRN-28-10 on CIFAR-10 (with dropout)
python cifar.py -a wrn --depth 28 --widen-factor 10 --drop 0.3 \
    --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 \
    --checkpoint checkpoints/cifar10/WRN-28-10-drop

# Evaluate a checkpoint
python cifar.py -a wrn --depth 28 --widen-factor 10 -e \
    --resume checkpoints/cifar10/WRN-28-10-drop/model_best.pth.tar
```

---

## Summary for Neural Race Multiview Project

This repository provides clean, battle-tested implementations of ResNet and WRN that could extend the neural-race-multiview experiments from synthetic data to real CIFAR data. Key adaptations needed:

1. **PyTorch API updates** (~10 lines to fix deprecated calls)
2. **Hook infrastructure** for pathway strength measurement (add activation/gradient hooks)
3. **KD training mode** for knowledge distillation experiments
4. **Multi-view data pipeline** (or use standard CIFAR with view-like augmentations)

The WRN-28-10 configuration is particularly relevant as it matches architectures commonly used in KD literature and provides sufficient capacity for multi-view feature learning.
