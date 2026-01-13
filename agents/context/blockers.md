# Blockers

Known issues preventing progress.

## Active Blockers

| ID | Description | Blocking | Owner | Status |
|----|-------------|----------|-------|--------|
| BLOCKER-001 | neural-race-multiview: Theory-Experiment Mismatch | Paper development | orchestrator | Under review |

### BLOCKER-001: neural-race-multiview Theory-Experiment Mismatch

- **Paper**: neural-race-multiview
- **Severity**: HIGH
- **Description**: Winner-take-all race dynamics not observed in experiments. Networks learn all views (coverage = 1.0) instead of single view (predicted coverage ≈ 0.33 for M=3 views).
- **Impact**: Core Theorem 2 (Race Dynamics) not validated; paper cannot proceed without resolution. Affects fundamental thesis of the work.
- **Root Cause**: Orthogonal slot structure in synthetic data may eliminate genuine competition between views. Views occupy disjoint dimensions, so network can learn all simultaneously without interference.
- **Investigated**:
  - ✓ Competing views (shared dimensions) — No effect, dominance still ≈ 0.35
  - ✓ Bottleneck architectures — Internal specialization but equal output usage
  - ✓ Asymmetric initialization (100x bias) — Initial advantage (0.60) erodes to equilibrium (0.35)
- **Status**: Awaiting orchestrator decision on next approach
- **Related Files**:
  - `papers/neural-race-multiview/log/experiment_log.md` (CRITICAL FINDING section)
  - `papers/neural-race-multiview/log/historical/DRAFT_next_steps_pending_review.md` (proposed solutions)
  - `papers/neural-race-multiview/log/experiment_findings/` (detailed reports)
- **Related Task**: TASK-001-neural-race-multiview-resolve-theory-mismatch

## Resolved Blockers

<!-- Move resolved blockers here for reference -->
