# Idea: Self-Distillation as Iterative Structure Amplification

- **ID**: IDEA-010
- **Stage**: developing
- **Created**: 2026-01-13
- **Promoted**: 2026-01-13
- **Source Papers**: PAPER-004 (Finzi et al., Epiplexity)
- **Related Work**: Allen-Zhu & Li (ICLR 2023), Pareek et al. (NeurIPS 2024), Furlanello et al. (ICML 2018)

## Research Question

**Does self-distillation work by iteratively increasing the accessibility of learned structure (P*), and can epiplexity estimates track and predict this process?**

More specifically:
1. Does the epiplexity of soft labels increase with each self-distillation round?
2. Does epiplexity saturate when performance saturates?
3. Can we predict the optimal number of self-distillation rounds from epiplexity dynamics?

## Hypothesis

**Primary hypothesis**: Self-distillation improves performance by re-encoding learned structure (P*) in progressively more extractable forms. Each round makes P* more accessible to gradient-based learning, until reaching a saturation point determined by model capacity.

**Quantitative prediction**:
```
d(S_T(soft_labels))/d(round) → 0  ⟺  d(performance)/d(round) → 0
```

Where S_T is epiplexity estimated via prequential coding.

**Mechanistic story**:
- **Round 0**: Model trained on hard labels extracts P* from raw data, encodes in weights
- **Round 1**: Soft labels encode P* explicitly in probability vectors → more accessible → model re-learns P* more completely
- **Round N**: Each round amplifies accessibility until P* is maximally extractable given architecture constraints

**Corollary predictions**:
1. Per-round performance gain correlates with per-round epiplexity increase
2. Saturation round correlates with model capacity (larger models saturate later)
3. Temperature affects amplification rate (higher τ → faster saturation but lower ceiling)

## Related Work

### Empirical Discovery of Self-Distillation

**Born-Again Networks** (Furlanello et al., ICML 2018)
- First systematic study of self-distillation with identical architectures
- Shows BANs outperform teachers on CIFAR-10 (3.5% error) and CIFAR-100 (15.5%)
- No theoretical explanation provided
- https://arxiv.org/abs/1805.04770

### Theoretical Explanations (Existing)

**1. Multi-View Feature Learning** (Allen-Zhu & Li, ICLR 2023 — Outstanding Paper Honorable Mention)
- Most comprehensive theory to date
- Key insight: Under "multi-view" data structure, single training learns only subset of features
- Self-distillation = implicit ensemble + distillation → learns more features
- Teacher's soft outputs contain "dark knowledge" about features student would miss
- Limitation: Explains *why* SD helps, not *how much* or *when to stop*
- https://arxiv.org/abs/2012.09816

**2. Variance Reduction in Linear Regression** (Pareek et al., NeurIPS 2024)
- Analyzes repeated SD in linear regression setting
- Shows r-step SD can reduce excess risk by factor of d (input dimension) vs 1-step
- Mechanism: ξ parameters control eigenvalue spectrum of estimator
- Limitation: Linear setting only; doesn't extend to deep networks directly
- https://arxiv.org/abs/2407.04600

**3. Regularization Amplification in Hilbert Space** (Mobahi et al., NeurIPS 2020)
- Models SD as iteratively limiting basis functions in RKHS
- Few rounds reduce overfitting; many rounds cause underfitting
- Predicts optimal stopping but doesn't provide measurable criterion
- https://proceedings.neurips.cc/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html

**4. Instance-Specific Label Smoothing** (Zhang et al., NeurIPS 2020)
- Views SD as amortized MAP estimation
- Soft labels provide instance-specific regularization on softmax outputs
- Connects SD to label smoothing literature
- https://proceedings.neurips.cc/paper/2020/file/1731592aca5fb4d789c4119c65c10b4b-Paper.pdf

**5. Loss Landscape Geometry** (Stanton et al., 2022)
- Identifies contradictions in prior explanations
- Proposes geometric view via loss landscape structure
- https://arxiv.org/abs/2206.08491

### Gap in Literature

| Existing Approaches | What They Explain | What They Don't |
|---------------------|-------------------|-----------------|
| Multi-view (Allen-Zhu) | WHY SD helps (more features) | How much gain? When to stop? |
| Variance (Pareek) | Multiplicative gain in linear case | Extension to deep networks |
| Regularization (Mobahi) | Overfitting reduction | Measurable stopping criterion |
| Label smoothing (Zhang) | Connection to regularization | Dynamics across rounds |

**No existing work uses:**
- Information-theoretic measures to track SD dynamics
- Epiplexity or computational MDL framing
- Measurable quantities that predict performance trajectory

## Why Novel

**Core differentiation**: Existing theories explain *why* self-distillation helps. IDEA-010 proposes *how to measure and predict* SD dynamics via epiplexity.

| Aspect | Prior Work | This Proposal |
|--------|------------|---------------|
| Question | Why does SD improve accuracy? | How can we track/predict SD gains? |
| Framework | Multi-view, regularization, variance | Information-theoretic (epiplexity) |
| Measurable? | No (implicit feature counting) | Yes (prequential coding estimate) |
| Predictive? | Qualitative (helps vs doesn't) | Quantitative (saturation point) |
| Stopping criterion | None or architecture-specific | Epiplexity saturation |

**Novel contributions**:

1. **First information-theoretic analysis of SD**: Frames self-distillation as iterative structure amplification using epiplexity (computational MDL)

2. **Measurable dynamics**: Proposes prequential epiplexity as a trackable quantity across SD rounds — no prior work provides this

3. **Predictive framework**: Hypothesis that epiplexity saturation predicts performance saturation could enable principled stopping rules

4. **Bridges two literatures**: Connects Finzi et al.'s epiplexity framework (2026) with SD theory — entirely unexplored connection

5. **Complements existing theories**: Not contradicting Allen-Zhu's multi-view theory; provides an *information-theoretic lens* on the same phenomenon (learning more features = extracting more structure = higher epiplexity)

## Potential Experiments

### Experiment 1: Core Hypothesis Test
**Goal**: Verify epiplexity tracks self-distillation dynamics

**Setup**:
- Dataset: CIFAR-100 (sufficient complexity, tractable)
- Architecture: ResNet-18 (standard, well-understood)
- Rounds: N = 10 (enough to observe saturation)
- Temperature: τ = 4 (standard KD temperature)

**Protocol**:
1. Train base model (round 0) on hard labels
2. For rounds 1-10:
   - Generate soft labels from previous round's model
   - Train new model on soft labels
   - Record: test accuracy, training loss curve
3. Compute epiplexity estimate per round:
   - Prequential method: area between learning curve and converged loss
   - Normalize by dataset size for comparability

**Metrics**:
- Test accuracy per round
- Epiplexity estimate per round
- Correlation(Δaccuracy, Δepiplexity) across rounds

**Success criteria**:
- Correlation > 0.7 between epiplexity and performance curves
- Both saturate within 2 rounds of each other

### Experiment 2: Capacity Dependence
**Goal**: Test if model capacity affects saturation point

**Setup**:
- Dataset: CIFAR-100
- Architectures: ResNet-{10, 18, 34, 50}
- Rounds: N = 10

**Protocol**: Same as Exp 1, but sweep architectures

**Prediction**: Larger models saturate later (more capacity to absorb structure)

**Metrics**:
- Saturation round per architecture
- Final epiplexity per architecture
- Correlation with parameter count / effective capacity

### Experiment 3: Temperature Effects
**Goal**: Test if temperature affects amplification dynamics

**Setup**:
- Dataset: CIFAR-100
- Architecture: ResNet-18
- Temperatures: τ ∈ {1, 2, 4, 8, 16}
- Rounds: N = 10

**Prediction**:
- Higher τ → faster epiplexity increase (more structure exposed per round)
- Higher τ → lower final epiplexity ceiling (lossy compression)
- Optimal τ balances rate vs. ceiling

**Metrics**:
- Epiplexity trajectory per temperature
- Final performance per temperature
- Rate of epiplexity increase (slope in early rounds)

### Experiment 4: Comparison to Standard KD
**Goal**: Differentiate self-distillation from teacher-student KD

**Setup**:
- Teacher: ResNet-50 (larger)
- Student: ResNet-18
- Compare: (a) self-distillation of ResNet-18, (b) KD from ResNet-50 to ResNet-18

**Prediction**:
- KD from larger teacher → higher epiplexity ceiling (teacher has richer P*)
- Self-distillation → slower but still measurable epiplexity increase

**Metrics**:
- Epiplexity trajectories for both settings
- Final performance comparison

## Open Questions

1. **What determines the saturation point?**
   - Model capacity? Architecture inductive biases? Dataset complexity?

2. **Is there a closed-form relationship between capacity and optimal rounds?**
   - Could enable predicting when to stop without running all rounds

3. **Does the finding generalize beyond classification?**
   - Self-distillation in generative models, regression, NLP?

4. **What happens if you distill past saturation?**
   - Performance degrades? Stays flat? Overfits to soft label noise?

5. **Can we accelerate structure amplification?**
   - Modified training procedures that increase epiplexity faster?

6. **Relationship to label smoothing?**
   - Label smoothing is "one-shot" structure injection; self-distillation is iterative

7. **Connection to lottery ticket hypothesis?**
   - Does self-distillation find/reinforce winning tickets?

## Promotion Criteria

- [x] Clear research question
- [x] Testable hypothesis with predicted outcome
- [x] Novelty argument (why this isn't already done)
- [x] Viable experiment plan (low compute, provable on small data)
- [x] Literature search complete — **confirmed no prior epiplexity + SD work** (2026-01-13)
- [ ] Preliminary evidence (run Experiment 1)

## Next Steps to Ready

1. ~~**Literature search**: Verify no prior work combining epiplexity with self-distillation analysis~~ ✓ Done
2. **Pilot experiment**: Run Experiment 1 on CIFAR-10 (faster iteration)
3. **Refine epiplexity estimation**: Implement prequential coding, validate on known cases
4. **If pilot succeeds**: Full Experiment 1-3 on CIFAR-100, draft results section

## Key Papers to Cite

- Finzi et al. (2026) — Epiplexity framework (arXiv:2601.03220)
- Allen-Zhu & Li (ICLR 2023) — Multi-view SD theory (arXiv:2012.09816)
- Pareek et al. (NeurIPS 2024) — Repeated SD gains (arXiv:2407.04600)
- Furlanello et al. (ICML 2018) — Born-Again Networks (arXiv:1805.04770)
- Mobahi et al. (NeurIPS 2020) — Regularization amplification
- Zhang et al. (NeurIPS 2020) — Instance-specific label smoothing
