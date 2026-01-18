# Summary: PyTorch ResNet CIFAR10

- **Repo ID**: REPO-003
- **GitHub**: https://github.com/akamaster/pytorch_resnet_cifar10
- **Paper**: He et al. 2015 - Deep Residual Learning for Image Recognition (arXiv:1512.03385)
- **Reviewed**: 2026-01-18

## One-Line Summary

A faithful PyTorch implementation of ResNet for CIFAR10 matching the original paper's architecture (Option A shortcuts with zero-padding), including pretrained models achieving state-of-the-art accuracy.

## Codebase Overview

### Structure
```
pytorch-resnet-cifar10/
├── resnet.py           # Model architecture (BasicBlock, ResNet class)
├── trainer.py          # Training loop, data loading, optimization
├── hubconf.py          # PyTorch Hub integration
├── run.sh              # Batch training script for all variants
├── pretrained_models/  # Pre-trained checkpoints (~95MB total)
│   ├── resnet20-12fca82f.th   (1.1MB, 0.27M params)
│   ├── resnet32-d509ac18.th   (1.9MB, 0.46M params)
│   ├── resnet44-014dd654.th   (2.7MB, 0.66M params)
│   ├── resnet56-4bfd9763.th   (3.5MB, 0.85M params)
│   ├── resnet110-1d1ed7c2.th  (7.0MB, 1.7M params)
│   └── resnet1202-f3b1deed.th (79MB, 19.4M params)
├── LICENSE             # BSD-2-Clause
└── README.md           # Documentation
```

### Tech Stack
- Language: Python 3.x
- Framework: PyTorch (any recent version)
- Key dependencies: torchvision (for CIFAR10 dataset only)

### Size & Complexity
- Lines of code: ~470 Python LOC (159 + 306 + 5)
- Core modules: 3 Python files
- Complexity assessment: **simple** - straightforward, minimal abstractions

## Quality Assessment

### Code Quality
- **Readability**: excellent - clean, well-commented, follows PyTorch conventions
- **Documentation**: adequate - good README, inline comments explain CIFAR-specific choices
- **Tests**: minimal - only a parameter counting test in resnet.py
- **Reproducibility**: easy - clear hyperparameters, pretrained models, documented training recipe

### Utility Assessment
- **Reusable components**:
  - ResNet architecture (resnet20/32/44/56/110/1202)
  - BasicBlock with Option A/B shortcuts
  - Training loop template
  - Pretrained checkpoints
- **Adaptation difficulty**: trivial - clean architecture, easy to modify
- **Active maintenance**: sporadic (last commit 2021, but stable code)

## Key Implementation Details

### ResNet Architecture (CIFAR-Specific)
**Location**: `resnet.py:L86-117`

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
```

**Key Insights**:
- **Initial conv**: 3x3 with stride 1 (NOT 7x7 + maxpool like ImageNet ResNet)
- **Starts with 16 filters** (NOT 64 like ImageNet)
- **Three stages**: 16 -> 32 -> 64 channels
- **Global average pooling**: `F.avg_pool2d(out, out.size()[3])` at end
- **No stem downsampling**: Input 32x32 stays 32x32 through conv1

**Differences from ImageNet ResNet**:
| Aspect | CIFAR ResNet | ImageNet ResNet |
|--------|--------------|-----------------|
| Input conv | 3x3, stride 1 | 7x7, stride 2 |
| Max pool | None | 3x3, stride 2 |
| Initial filters | 16 | 64 |
| Stages | 3 | 4 |
| Channels | 16/32/64 | 64/128/256/512 |

### Option A Shortcuts (Zero-Padding)
**Location**: `resnet.py:L64-77`

```python
if stride != 1 or in_planes != planes:
    if option == 'A':
        """
        For CIFAR10 ResNet paper uses option A.
        """
        self.shortcut = LambdaLayer(lambda x:
                                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    elif option == 'B':
        self.shortcut = nn.Sequential(
             nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
             nn.BatchNorm2d(self.expansion * planes)
        )
```

**Key Insights**:
- **Option A**: Zero-padding for dimension matching (no extra parameters)
  - `x[:, :, ::2, ::2]` downsamples spatially by 2
  - `F.pad(..., planes//4, planes//4)` pads channels with zeros
- **Option B**: 1x1 projection convolution (adds parameters)
- **Default is Option A** - matches original CIFAR paper
- **Critical for theory**: Option A has no learnable parameters in shortcut; Option B adds projection weights

### BasicBlock Architecture
**Location**: `resnet.py:L54-83`

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # ... shortcut setup

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

**Key Insights**:
- **Pre-activation**: No, uses post-activation (ReLU after addition)
- **Bias**: No bias in conv layers (absorbed by BatchNorm)
- **Two 3x3 convs** per block (basic block, not bottleneck)
- **ReLU after shortcut addition**: Creates nonlinearity at skip connection

### Weight Initialization
**Location**: `resnet.py:L39-43`

```python
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
```

**Key Insights**:
- **Kaiming He initialization** for conv and linear layers
- **No bias initialization** (biases disabled in convs)
- **BatchNorm uses defaults** (gamma=1, beta=0)

### Training Configuration
**Location**: `trainer.py:L109-127`

```python
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[100, 150], last_epoch=args.start_epoch - 1)
```

**Training Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 128 |
| Initial LR | 0.1 |
| LR schedule | MultiStepLR [100, 150] |
| LR decay | 0.1x at milestones |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |

**ResNet110/1202 Warmup**:
```python
if args.arch in ['resnet1202', 'resnet110']:
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr*0.1  # Start at 0.01 for first epoch
```

### Data Augmentation
**Location**: `trainer.py:L88-99`

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
```

**Key Insights**:
- **Normalization**: ImageNet statistics (NOT CIFAR10 statistics)
  - This is a minor deviation from best practices but works well
- **Random horizontal flip**: Standard augmentation
- **Random crop with padding 4**: Randomly crop 32x32 from 40x40 padded image
- **No color jitter, cutout, or other advanced augmentation**

### Pretrained Model Loading
**Location**: `trainer.py:L74-84`

```python
if args.resume:
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
```

**Checkpoint Structure**:
```python
{
    'epoch': int,
    'state_dict': OrderedDict,  # Model weights (with DataParallel wrapper)
    'best_prec1': float         # Best validation accuracy
}
```

**Key Insights**:
- Checkpoints saved with `nn.DataParallel` wrapper
- To load without DataParallel: strip `module.` prefix from keys
- Pretrained models are final training checkpoints, not EMA

## Architecture Notes

### Data Flow
```
Input (3, 32, 32)
    |
    v
conv1: 3x3, 16 filters, stride 1 -> (16, 32, 32)
bn1 + relu
    |
    v
layer1: n blocks, 16 filters -> (16, 32, 32)  [no spatial downsampling]
    |
    v
layer2: n blocks, 32 filters, stride 2 -> (32, 16, 16)  [downsample once]
    |
    v
layer3: n blocks, 64 filters, stride 2 -> (64, 8, 8)   [downsample once]
    |
    v
avg_pool: global -> (64, 1, 1)
    |
    v
linear: 64 -> 10 -> logits
```

### Layer Counts
| Model | Blocks [layer1, layer2, layer3] | Total Layers | Parameters |
|-------|----------------------------------|--------------|------------|
| ResNet20 | [3, 3, 3] | 20 | 0.27M |
| ResNet32 | [5, 5, 5] | 32 | 0.46M |
| ResNet44 | [7, 7, 7] | 44 | 0.66M |
| ResNet56 | [9, 9, 9] | 56 | 0.85M |
| ResNet110 | [18, 18, 18] | 110 | 1.7M |
| ResNet1202 | [200, 200, 200] | 1202 | 19.4M |

**Counting**: 1 (conv1) + 2n (layer1) + 2n (layer2) + 2n (layer3) + 1 (linear) = 6n + 2

### Design Decisions
1. **Option A by default** - Matches paper, fewer parameters, cleaner gradient flow
2. **No bottleneck blocks** - CIFAR images too small for bottleneck architecture
3. **Three stages only** - ImageNet has four; CIFAR only needs three
4. **Global average pooling** - Standard for classification, reduces parameters vs FC

---

## Connection to Saxe Neural Race Theory

### How ResNet Maps to GDLN Framework

The Saxe et al. 2022 paper shows ReLU networks are equivalent to Gated Deep Linear Networks:

$$f(x) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

where $D_\ell(x) = \text{diag}(G_\ell(x))$ are input-dependent binary gates.

**In ResNet, "pathways" manifest as**:
1. **Gating patterns**: Which ReLU neurons are active for a given input
2. **Skip connections**: Create additional pathways that bypass nonlinearities
3. **Residual blocks**: Each block has two "routes" - the skip path and the residual path

### Pathway Structure in BasicBlock

```
Input x
   |-------------------+
   |                   |
   v                   |
 conv1 -> bn1 -> ReLU  | shortcut (identity or pad)
   |                   |
   v                   |
 conv2 -> bn2          |
   |                   |
   +------- ADD -------+
            |
            v
          ReLU
            |
            v
         Output
```

**Two pathways per block**:
1. **Residual path**: conv1 -> ReLU -> conv2 (multiplicative, depends on ReLU gates)
2. **Skip path**: identity or zero-pad (linear, always active)

For a ResNet20 with 9 blocks total, there are theoretically $2^9 = 512$ distinct "pathways" through the network, though not all are equally important.

### Where Would "Race Dynamics" Appear?

Under the Saxe theory, pathways compete through the saturation mechanism. In ResNet:

1. **Competing pathways for a feature**: If multiple ReLU patterns can detect the same feature, they may "race" during early training
2. **Skip vs residual competition**: At each block, the network learns how much to use skip vs residual path
3. **Layer-wise feature emergence**: Different layers might compete to learn certain features

**However**, critical differences from GDLN:
- **Loss function**: This implementation uses CrossEntropyLoss (theory requires MSE)
- **Optimization**: Uses SGD with momentum 0.9 (theory assumes gradient flow)
- **BatchNorm**: Changes gradient dynamics significantly
- **No explicit pathway tracking**: Can't directly measure pathway strengths

### Measuring Pathway Strengths in ResNet

To study race dynamics, we would need to measure pathway strength $s_{y,m}(t)$. Possible approaches:

1. **Gate pattern tracking**: Record which ReLUs are active for each class
   ```python
   # Hook to record activations
   def hook_fn(module, input, output):
       gate_pattern = (output > 0).float()
       return gate_pattern
   ```

2. **Gradient flow analysis**: Measure gradient magnitude through skip vs residual paths
   ```python
   # Register hooks on shortcut and residual paths
   def residual_grad_hook(grad):
       residual_grad_norms.append(grad.norm().item())
   ```

3. **Singular value decomposition**: Track SVD of layer weights per class
   ```python
   # For class-conditional pathway strength
   W_effective = W2 @ diag(G) @ W1  # Per gating pattern
   U, S, V = torch.svd(W_effective)
   pathway_strength = S[0]  # Largest singular value
   ```

4. **Ablation-based measurement**: Measure accuracy drop when blocking specific paths

### Modifications for Neural Race Experiments

To study race dynamics using this codebase:

**Minimal modifications**:
```python
# 1. Switch to MSE loss for regression setup
criterion = nn.MSELoss()
# Need one-hot targets
target_onehot = F.one_hot(target, num_classes=10).float()

# 2. Reduce/remove momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

# 3. Add pathway strength tracking
class PathwayTracker:
    def __init__(self, model):
        self.gate_patterns = {}
        self._register_hooks(model)

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(
                    lambda m, i, o, n=name: self._record_gate(n, o)
                )

    def _record_gate(self, name, output):
        self.gate_patterns[name] = (output > 0).float().mean(dim=0)
```

**Structural modifications**:
```python
# 4. Option to remove BatchNorm (cleaner gradient dynamics)
class BasicBlockNoBN(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        # No BatchNorm layers
```

---

## Relevance to Lab Research

### Potential Reuse
- [x] **Teacher models for KD**: Pretrained ResNet56/110 as strong teachers
- [x] **Baseline architecture**: Clean CIFAR ResNet for comparative experiments
- [x] **Training recipe**: Standard hyperparameters for CIFAR training
- [ ] **Direct pathway measurement**: Requires modification (no built-in tracking)
- [ ] **Race dynamics study**: Requires loss/optimizer changes per PAPER-001

### Learnings
1. **Option A shortcuts preserve gradient flow**: Zero-padding doesn't add parameters or nonlinearity to skip path
2. **BatchNorm complicates theory**: Running statistics add non-gradient-descent dynamics
3. **Momentum breaks gradient flow assumptions**: The 0.9 momentum means current gradients don't directly drive updates
4. **CrossEntropyLoss saturation differs from MSE**: CE saturates differently, may not create same competition

### Gaps or Limitations
1. **No pathway tracking built in**: Need to add hooks for race dynamics measurement
2. **Uses CrossEntropyLoss**: Theory requires MSE for exact race dynamics
3. **High momentum (0.9)**: Violates gradient flow assumption
4. **BatchNorm**: Changes dynamics vs pure GDLN theory
5. **No multi-view setup**: Single canonical view per image

### Integration with neural-race-multiview

**For KD experiments**:
```python
# Load pretrained teacher
teacher = resnet.resnet56()
checkpoint = torch.load('pretrained_models/resnet56-4bfd9763.th')
# Strip DataParallel prefix if needed
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
teacher.load_state_dict(state_dict)
teacher.eval()

# KD loss
def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.9):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**For multi-view experiments on real CIFAR**:
```python
# Create multiple "views" via augmentation
class MultiViewTransform:
    def __init__(self, num_views=2):
        self.transforms = [
            transforms.Compose([transforms.RandomHorizontalFlip(), ...]),
            transforms.Compose([transforms.ColorJitter(...), ...]),
        ]

    def __call__(self, x):
        return [t(x) for t in self.transforms]
```

## Cross-References
- **Paper (He 2015)**: arXiv:1512.03385 (original ResNet paper)
- **Related Repos**: REPO-001 (gated-dln), REPO-002 (pytorch-classification)
- **Lab Papers Using**: papers/neural-race-multiview/
- **Paper Summary**: reading_stack/summaries/PAPER-001-saxe-2022.md (theoretical foundation)
