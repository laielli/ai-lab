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
- **Status**: REQUIRES PAPER DIRECTION DECISION
- **Options**:
  1. **Revise theory** — Competition is designed, not emergent; modify Theorem 3
  2. **Find emergent competition** — Different architectures (transformers, attention)
  3. **Reframe contribution** — Controlled study of competition mechanisms
- **Related Files**:
  - `papers/neural-race-multiview/log/experiment_log.md` (Full findings with conclusions)
  - `papers/neural-race-multiview/src/train.py` (train_with_competition function)
  - `papers/neural-race-multiview/src/model.py` (DeepLinearNet)
- **Related Task**: TASK-001-neural-race-multiview-resolve-theory-mismatch

## Resolved Blockers

<!-- Move resolved blockers here for reference -->
