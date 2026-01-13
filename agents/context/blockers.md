# Blockers

Known issues preventing progress.

## Active Blockers

| ID | Description | Blocking | Owner | Status |
|----|-------------|----------|-------|--------|
| BLOCKER-001 | neural-race-multiview: Theory-Experiment Mismatch | Paper development | orchestrator | DECISION REQUIRED |

### BLOCKER-001: neural-race-multiview Theory-Experiment Mismatch

- **Paper**: neural-race-multiview
- **Severity**: HIGH
- **Description**: Winner-take-all race dynamics not observed in experiments. Networks learn all views (coverage = 1.0) instead of single view (predicted coverage ≈ 0.33 for M=3 views).
- **Impact**: Core Theorem 2 (Race Dynamics) and Theorem 3 (KD Mechanism) not validated as originally stated. Requires paper pivot decision.
- **Root Cause (UPDATED 2026-01-13)**: Standard neural network training does NOT naturally produce winner-take-all dynamics. The theory's saturation mechanism (s_max) doesn't emerge from discrete SGD.
- **Investigated**:
  - ✓ Competing views (shared dimensions) — No effect, dominance still ≈ 0.35
  - ✓ Bottleneck architectures — Internal specialization but equal output usage
  - ✓ Asymmetric initialization (100x bias) — Initial advantage (0.60) erodes to equilibrium (0.35)
  - ✓ Deep linear networks with MSE loss — Still learns all views
  - ✓ Gradient flow (no momentum) — Still learns all views
  - ✓ **Explicit competition loss — WORKS! Dominance ≈ 0.6**
  - ✓ Capacity constraints (hidden=1-50) — Reduces coverage but not WTA
  - ✓ Weight decay (L2) — Model collapse at high values, no WTA
  - ✓ Entropy-based competition — Creates WTA (dominance ≈ 0.55)
- **Key New Findings (2026-01-13)**:
  1. Competition requires **explicit loss term** — doesn't emerge from architecture/capacity
  2. KD does NOT break competition — contradicts Theorem 3 prediction
  3. KD performs WORSE under competition — soft labels weaker than hard labels
- **BREAKTHROUGH (2026-01-13 - GatedDLN Analysis)**:
  - Analyzed Facebook Research's gated-dln implementation
  - **Key insight**: Saxe theory's "race" is about **singular value competition in SVD space**, NOT view competition
  - GatedDLN architecture has explicit pathway separation (M encoders, M decoders, binary gate)
  - Implemented GatedDLN for our setup and confirmed: **race dynamics ARE happening in SVD space**
  - Different singular value modes grow at different rates matching input-output correlations
  - Example: Target SVs [13.60, 6.40, 3.00, 3.00] → Growth ratios [8.68, 6.78, 5.34, 5.31]
  - This explains why standard MLPs don't show WTA: they lack explicit pathway separation
- **KD + GatedDLN Experiment (2026-01-13)**:
  - Tested Theorem 3 with GatedDLN on structured regression
  - **FINDING: KD does NOT break race dynamics**
  - Hard labels dominance: 0.505, KD dominance: 0.502 (essentially identical)
  - Both methods converge to same relative pathway proportions
  - Race is determined by data correlations, not training method
- **Status**: REQUIRES THEORY REVISION - race dynamics confirmed but KD doesn't preserve diversity
- **Options** (Updated with new understanding):
  1. **Use GatedDLN architecture** — Theory validated with proper architecture; race happens in SVD space
  2. **Reframe paper scope** — Theory applies to architectures with explicit pathway separation, not standard MLPs
  3. **Bridge theory-practice gap** — Document when/how Saxe theory applies to different architectures
- **Related Files**:
  - `papers/neural-race-multiview/log/experiment_log.md` (Full findings with conclusions)
  - `papers/neural-race-multiview/src/train.py` (train_with_competition, train_gated_dln functions)
  - `papers/neural-race-multiview/src/model.py` (DeepLinearNet, GatedDLN, GatedMultiViewNet)
  - `papers/neural-race-multiview/src/experiments/exp_gated_dln.py` (GatedDLN experiments)
  - `code_stack/summaries/REPO-001-gated-dln.md` (Analysis of Facebook Research implementation)
- **Related Task**: TASK-001-neural-race-multiview-resolve-theory-mismatch

## Resolved Blockers

<!-- Move resolved blockers here for reference -->
