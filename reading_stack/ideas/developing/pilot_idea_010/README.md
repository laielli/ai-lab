# IDEA-010 Pilot Experiment

**Self-Distillation as Iterative Structure Amplification**

Tests whether epiplexity (learnable structure) tracks self-distillation dynamics on CIFAR-10.

## Hypothesis

```
d(S_T(soft_labels))/d(round) → 0  ⟺  d(performance)/d(round) → 0
```

Epiplexity should saturate when performance saturates.

## Quick Start (Cloud GPU)

### Option 1: Lambda Labs / Vast.ai / RunPod

```bash
# 1. SSH into your GPU instance

# 2. Clone or copy this directory
git clone <repo-url>
cd ai-lab/reading_stack/ideas/developing/pilot_idea_010

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run experiment (SimpleCNN first - faster)
python run_pilot.py --model simple_cnn --rounds 5 --epochs 50

# 5. Run with ResNet-18 (validation)
python run_pilot.py --model resnet18 --rounds 5 --epochs 50
```

### Option 2: Google Colab

Copy and run in a Colab cell:

```python
!git clone <repo-url>
%cd ai-lab/reading_stack/ideas/developing/pilot_idea_010
!pip install -q torch torchvision matplotlib tqdm

# Run experiment
!python run_pilot.py --model simple_cnn --rounds 5 --epochs 50
```

## Expected Runtime

| Model | Device | 5 rounds × 50 epochs |
|-------|--------|---------------------|
| SimpleCNN | T4 GPU | ~10 min |
| SimpleCNN | A100 | ~5 min |
| ResNet-18 | T4 GPU | ~30 min |
| ResNet-18 | A100 | ~15 min |

## Files

- `run_pilot.py` - Main experiment script
- `models.py` - SimpleCNN and ResNet-18 definitions
- `train.py` - Hard label and soft label training loops
- `epiplexity.py` - Prequential epiplexity estimation
- `results/` - Output directory (JSON + plots)

## Output

Results saved to `results/`:
- `pilot_results_simple_cnn.json` - Full metrics and loss histories
- `pilot_results_simple_cnn.png` - Visualization plots
- `pilot_results_resnet18.json` - ResNet-18 results
- `pilot_results_resnet18.png` - ResNet-18 plots

## Success Criteria

| Metric | Target |
|--------|--------|
| Accuracy improves with SD | Round N > Round 0 |
| Epiplexity increases | Round N > Round 0 |
| Correlation(Δacc, Δepip) | > 0.5 |

## Command Line Options

```
python run_pilot.py --help

Options:
  --model {simple_cnn,resnet18}  Model architecture
  --rounds INT                   Self-distillation rounds (default: 5)
  --epochs INT                   Epochs per round (default: 50)
  --lr FLOAT                     Learning rate (default: 0.01)
  --temperature FLOAT            Distillation temperature (default: 4.0)
  --batch-size INT               Batch size (default: 128)
  --seed INT                     Random seed (default: 42)
  --output DIR                   Output directory (default: results)
  --device {cuda,cpu,mps}        Device (auto-detected if not specified)
```

## Retrieving Results

After running on cloud, download results:

```bash
# From cloud instance
scp -r results/ local_machine:~/pilot_results/

# Or use the JSON directly
cat results/pilot_results_simple_cnn.json | python -m json.tool
```
