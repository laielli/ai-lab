# Task: Resolve neural-race-multiview Theory-Experiment Mismatch

- **ID**: TASK-001
- **Owner**: orchestrator
- **Paper**: neural-race-multiview
- **Created**: 2026-01-13

## Description

Critical finding from completed experiments: Theory predictions don't match observed behavior.

**Expected** (from Theorem 2 - Race Dynamics):
- Winner-take-all dynamics
- Networks converge to single-view solutions
- Coverage ≈ 1/M (≈ 0.33 for M=3 views)
- One pathway dominates, others atrophy

**Observed** (from 11 completed experiments):
- All views learned equally
- Coverage = 1.0 (all views captured)
- All pathways grow with equal strength
- Dominance ≈ 0.34 (no clear winner)

**Root Cause Hypothesis**: Orthogonal slot structure in synthetic data eliminates genuine competition. Views occupy disjoint dimensions, allowing network to learn all simultaneously without interference.

Engineer has investigated multiple approaches without success:
- Competing views with shared dimensions → Still no winner-take-all (dominance = 0.35)
- Bottleneck architectures → Internal specialization but equal output
- Asymmetric initialization (100x bias) → Advantage erodes during training (0.60 → 0.35)

Advisor has drafted next steps document awaiting review (see handoff notes).

## Acceptance Criteria

- [ ] Review advisor's DRAFT_next_steps_pending_review.md proposal
- [ ] Decide on path forward:
  - Option A: Modify experimental setup (stronger competition, different architecture)
  - Option B: Revise theory to account for orthogonal case
  - Option C: Add assumptions/scope to theory (e.g., "competing views only")
  - Option D: Other approach
- [ ] Document decision and rationale
- [ ] Create follow-up tasks for chosen approach
- [ ] Update BLOCKER-001 status in agents/context/blockers.md

## Handoff Notes

**From**: Engineer (via experiment reports, 2026-01-10)

**Context**: P1 experiments completed. All infrastructure working correctly. Results are reproducible and consistent across multiple runs. The issue is genuine theory-experiment mismatch, not implementation bug.

**Key Files**:
- `papers/neural-race-multiview/log/experiment_log.md` - Comprehensive experiment log with CRITICAL FINDING section
- `papers/neural-race-multiview/log/historical/DRAFT_next_steps_pending_review.md` - Advisor's proposed solutions
- `papers/neural-race-multiview/log/experiment_findings/*.md` - Detailed experiment reports
- `papers/neural-race-multiview/log/results/` - Raw experimental data (JSON + figures)

**Experiments Completed** (11/17):
- exp_2_1_single_view.py - Single-view convergence test
- exp_2_3_race_dynamics.py - Pathway competition visualization
- exp_3_1_kd_coverage.py - KD coverage transfer
- exp_3_3_pathway_evolution.py - Pathway survival comparison
- exp_competing_views.py - Shared dimensions test
- exp_bottleneck.py - Capacity constraint test
- exp_asymmetric_init.py - Initial advantage test
- Plus 4 more follow-up investigations

**Impact**: This is a HIGH severity blocker. Core Theorem 2 (Race Dynamics) is not validated. Paper cannot proceed to NeurIPS submission without resolution. May require fundamental revision to theory or significant changes to experimental methodology.

---

## Notes

This task is critical for paper development. All other work on neural-race-multiview is blocked pending resolution. Target: Resolve within 1-2 weeks to stay on track for May deadline.
