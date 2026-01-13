# Summary: Gated Deep Linear Networks (gated-dln)

- **Repo ID**: REPO-001
- **GitHub**: https://github.com/facebookresearch/gated-dln
- **Paper**: PAPER-001 (Saxe et al. 2022 - Neural Race Reduction)
- **Reviewed**: 2026-01-13

## One-Line Summary

Official implementation of Neural Race Reduction theory for Gated Deep Linear Networks, with **critical gap**: experiments use CrossEntropyLoss while theory uses MSELoss.

## Codebase Overview

### Structure
```
gated-dln/
├── config/              # Hydra configuration files
│   ├── model/          # k_path.yaml model config
│   ├── optimizer/      # SGD, Adam configs
│   └── experiment/     # k_path experiment config
├── notebooks/          # Theory verification (MSE loss)
│   ├── Gated_DLN_solution_verification_figure.ipynb
│   └── Gated_DLN_vs_ntk_figure.ipynb
├── scripts/            # mnist.sh, cifar10.sh
├── src/
│   ├── model/          # k_path.py - main model
│   ├── experiment/     # Training loops
│   ├── task/           # Multi-view data generation
│   └── checkpoint/     # Checkpointing utilities
├── tests/              # pytest suite
└── main.py             # Hydra entry point
```

### Tech Stack
- Language: Python 3.9
- Framework: PyTorch 1.11.0
- Key dependencies:
  - Hydra 1.2.0 (configuration)
  - functorch 0.1.1 (batched vmap operations)
  - ml-moe 0.3 (mixture of experts layers)
  - xplogger 0.10.0 (experiment logging)

### Size & Complexity
- Lines of code: ~4,800 Python LOC
- Core modules: 61 Python files in src/
- Notebooks: 2 Jupyter notebooks (~88% of repo by size)
- Complexity assessment: **moderate** - clean architecture but dense with configuration

## Quality Assessment

### Code Quality
- **Readability**: fair - type hints present but some confusing method swaps at runtime
- **Documentation**: minimal - sparse docstrings, relies on code clarity
- **Tests**: minimal - parametric tests exist but only check training runs, not correctness
- **Reproducibility**: possible - configs provided but requires FAIR cluster infrastructure

### Utility Assessment
- **Reusable components**:
  - Gating mechanism (`make_gate()` in k_path.py)
  - Multi-view transforms (rotation, permutation)
  - Pathway tracking metrics
- **Adaptation difficulty**: significant - tightly coupled to Hydra config system
- **Active maintenance**: abandoned (archived October 2023)

## Key Implementation Details

### CRITICAL: Loss Function Mismatch
**Location**: `src/model/k_path.py:L114`

```python
self.loss_fn = nn.CrossEntropyLoss(reduction="none")
```

**Summary**: The main experiment code uses **CrossEntropyLoss**, NOT MSELoss as the paper's theory requires. However, the theory verification notebooks correctly use MSELoss.

**Key Insights**:
- Paper theory (equations, gradient flow) assumes MSE loss
- Main experiments train with cross-entropy for classification
- Notebooks (`Gated_DLN_solution_verification_figure.ipynb`) use MSE and match theory

**CRITICAL Difference from Paper**: This is the smoking gun for why winner-take-all dynamics may not appear in classification experiments. The theory's race dynamics depend on MSE's specific gradient structure.

### Gating Mechanism
**Location**: `src/model/k_path.py:L144-178`

**Summary**: Binary gate matrix controls which encoder-decoder paths are trained. Gate is fixed (not learned).

```python
def make_gate(self) -> torch.Tensor:
    if self.gate_cfg["mode"] == "fully_connected":
        gate = torch.ones(*self.tasks.shape, device="cpu", dtype=torch.float32)
        return gate
    # For modes like "10_plus_mod", creates band-diagonal structure
    input_output_map = self._get_input_output_map(mode=self.gate_cfg["mode"])
    gate = torch.zeros(*self.tasks.shape, device="cpu", dtype=torch.float32)
    for current_input, current_output in input_output_map:
        gate[current_input][current_output] = 1.0
```

**Key Insights**:
- Gate modes: `mod`, `k_plus_mod`, `k_plus_minus_mod`, `fully_connected`
- `k_plus_mod`: Each encoder connects to k decoders in cyclic pattern
- Gate is **fixed at initialization** - satisfies theory's Assumption A3
- `permute` suffix randomly permutes column order

### Multi-View Data Generation
**Location**: `src/task/transforms/input.py:L17-72`

**Summary**: Two methods for creating "views" - rotation and permutation transforms.

```python
def get_list_of_rotation_transformations(num_transformations, full_angle=180.0):
    transforms = []
    for input_index in range(num_transformations):
        angle = full_angle * input_index / num_transformations
        transforms.append(get_rotation_transform(angle=angle))
    return TransformList(transforms)

def get_list_of_permutation_transformations(dataset_name, num_transformations, device):
    transforms = []
    for input_index in range(num_transformations):
        transforms.append(get_permutation_transform(
            dataset_name=dataset_name,
            seed=input_index,  # Deterministic per transform
            device=device,
        ))
    return TransformList(transforms)
```

**Key Insights**:
- Rotations span 0 to `full_angle` degrees (270 for MNIST)
- Permutations use seeded RNG for reproducibility
- Views are NOT orthogonal in general - just different transforms
- Theory assumes orthogonal views; implementation doesn't enforce this

### Pathway Strength Tracking
**Location**: `src/experiment/k_path.py:L166-201`

**Summary**: Accuracy and loss tracked separately for gated (selected) and ungated paths.

```python
gate = self.model.gate
gate_sum = gate.sum()
flipped_gate = (gate == 0).float()
flipped_gate_sum = flipped_gate.sum()

average_accuracy_for_selected_paths = (num_correct * gate).sum() / (gate_sum * total)
average_accuracy_for_unselected_paths = (num_correct * flipped_gate).sum() / (flipped_gate_sum * total)
```

**Key Insights**:
- No explicit $s_{y,m}(t)$ pathway strength computation
- Only binary: selected paths vs unselected paths
- Missing: SVD-based pathway strength as in theory
- Missing: Continuous strength evolution tracking

### Theory Verification (Notebooks)
**Location**: `notebooks/Gated_DLN_solution_verification_figure.ipynb`

**Summary**: Notebooks implement clean GatedDLN with **MSE loss** and verify against theoretical ODE predictions.

```python
# From notebook - correct theory verification setup
cfg.loss = {"cls": "torch.nn.MSELoss", "reduction": "none"}
cfg.optimizer = {"cls": "torch.optim.SGD", "lr": .02, "momentum": 0, ...}
```

**Key Insights**:
- Uses MSE loss (matching theory)
- Uses SGD with momentum=0 (approximating gradient flow)
- Tracks SVD of weights: `input_svs`, `hidden_svs`, `out_svs`
- Theory ODE predicts evolution: `b1, b2, b3` singular values
- Shows excellent theory-experiment match **when using MSE**

### Key Experimental Configurations
**Location**: `scripts/mnist.sh`, `scripts/cifar10.sh`

```bash
# MNIST script parameters
experiment.num_epochs=1000
experiment.task.num_input_transformations=100  # 100 views
experiment.task.num_classes_in_selected_dataset=10
model.hidden_layer_cfg.dim=128,1024
model.should_use_non_linearity=False  # Linear hidden layers!
model.weight_init.gain=10.0,1.0,0.1,0.01,0.001,0.0001
optimizer=sgd
optimizer.lr=0.0001
optimizer.momentum=0.9  # Non-zero momentum
```

**Key Insights**:
- Linear hidden layers (`should_use_non_linearity=False`)
- Wide range of initialization scales tested
- Uses momentum (deviates from gradient flow)
- Very small learning rate (0.0001)

## Architecture Notes

### Data Flow
```
Input X -> [Transform_i] -> Encoder_i -> Shared Hidden -> Decoder_j -> Output
                                                |
                                      Gate[i,j] masks loss
```

1. Input transformed by M different transforms (rotation/permutation)
2. Each transform has dedicated encoder pathway
3. All encoders feed through **shared** hidden layer
4. M decoders (one per output task)
5. Gate matrix selects which (encoder, decoder) pairs contribute to loss

### Key Abstractions
- `TasksForKPathModel`: Container for transforms and task structure
- `Model (k_path.py)`: Main model with encoders, hidden, decoders, gate
- `Experiment (k_path.py)`: Training loop with metric tracking
- `TransformList`: Collection of input/output transforms

### Design Decisions
1. **Shared hidden layer required** (`hidden_layer_cfg.should_share=True`)
   - Critical for theory - enables pathway competition through shared representation

2. **Fixed gate at initialization**
   - Gate is not learned during training
   - Satisfies theory's Assumption A3 (fixed gating patterns)

3. **MoE-style decoders**
   - Uses mixture-of-experts layer structure for efficient decoder computation
   - Enables O(k) pathways with O(k^2) evaluation

## Relevance to Lab Research

### Potential Reuse
- [x] Can reuse gating mechanism for neural-race-multiview
- [x] Can adapt pathway tracking metrics (but need to add SVD tracking)
- [ ] Cannot directly reuse main training loop (wrong loss function)
- [x] Can reuse notebook's GatedDLN class as reference implementation

### Learnings
1. **Theory requires MSE loss** - the notebooks confirm this clearly
2. **No explicit saturation constraint** - $s_{max}$ is not implemented; theory assumes unbounded growth leading to winner-take-all
3. **Pathway strength not tracked as SVD** - experiments only track accuracy/loss, not singular values that theory predicts
4. **Init scale matters critically** - notebooks show how small vs large init changes dynamics (lazy vs feature learning regimes)

### Gaps or Limitations
1. **Major theory-experiment gap**: Main code uses CrossEntropyLoss; theory requires MSE
2. **No gradient flow mode**: Uses discrete SGD with momentum, not continuous gradient flow
3. **No saturation mechanism**: Theory's $s_{max}$ constraint not implemented
4. **Limited pathway tracking**: Binary selected/unselected, not continuous strength
5. **Views not orthogonal**: Permutation/rotation transforms don't guarantee orthogonality
6. **Archived/unmaintained**: No ongoing development since Oct 2023

## Critical Findings for BLOCKER-001

The neural-race-multiview project's theory-experiment mismatch (BLOCKER-001) likely stems from:

1. **Loss function**: Using CrossEntropyLoss instead of MSE fundamentally changes gradient dynamics
2. **Discrete SGD**: Theory assumes gradient flow; implementation uses discrete updates with momentum
3. **No saturation**: Theory predicts winner-take-all only when pathways can saturate; no such mechanism exists

**Recommendation**: To validate race dynamics, experiments should:
- Use MSE loss with regression targets
- Use SGD with very small learning rate and zero momentum
- Track pathway singular values, not just accuracy
- Implement explicit saturation constraint or verify it emerges naturally

## Cross-References
- **Paper Summary**: reading_stack/summaries/PAPER-001-saxe-2022.md
- **Related Repos**: None identified
- **Lab Papers Using**: papers/neural-race-multiview/
