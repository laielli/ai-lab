# Summary: Make Haste Slowly: A Theory of Emergent Structured Mixed Selectivity in Feature Learning ReLU Networks

- **Paper ID**: PAPER-002
- **arXiv**: 2503.06181
- **Authors**: Devon Jarvis, Richard Klein, Benjamin Rosman, Andrew M. Saxe
- **Venue**: ICLR 2025 (Accepted)
- **Summarized**: 2026-01-13

## Focus Area Tags
- Mechanistic DL Theory
- Feature Learning

## One-Line Summary

This paper establishes an equivalence between finite-width ReLU networks and Gated Deep Linear Networks (GDLNs), deriving that ReLU networks possess an inductive bias toward "structured mixed selectivity" representations rather than strictly modular ones, driven by node-reuse and learning speed optimization.

---

## Key Contributions

1. **ReLU-GDLN Equivalence for Feature Learning**: Proves that ReLU networks can be expressed as Rectified Linear Networks (ReLNs), specific GDLNs that replicate ReLU behavior at all training timesteps, enabling analytical treatment of finite-width ReLU dynamics.

2. **Structured Mixed Selectivity Theory**: Demonstrates that optimal representations are not strictly modular/disentangled but exhibit "structured mixed selectivity" where pathways are active across multiple contexts while maintaining computational differentiation.

3. **Node-Reuse Bias Mechanism**: Shows that mixed selectivity emerges from an implicit bias toward node-reuse that maximizes learning speed by increasing effective dataset size per pathway.

4. **Scaling Laws for Mixed Selectivity**: Proves that mixed selectivity intensifies with (a) more contexts and (b) additional hidden layers.

---

## Methodology

### The ReLN-GDLN Equivalence

The key insight: ReLU nonlinearity can be expressed as implicit gating. A ReLU network:
$$f(x) = W_L \cdot \sigma(W_{L-1} \cdot \sigma(\cdots \sigma(W_1 x)))$$

becomes a Gated Deep Linear Network:
$$f(x) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

where $D_\ell(x) = \text{diag}(G_\ell(x))$ are diagonal matrices with binary gates $G_\ell(x) = \mathbb{1}[h_\ell(x) > 0]$.

A **Rectified Linear Network (ReLN)** is defined as "a GDLN with gates $G^*(x^i)$ such that GDLN output equals ReLU output for all training times and all datapoints."

**Proposition 4.2**: There is a unique ReLN (up to symmetries) such that GDLN loss equals ReLU loss for all training times, and this ReLN finds the **neural race winning strategy**.

### Multi-Task Contextual Setup

The task is reminiscent of multi-task learning:
- **Input**: Concatenation of one-hot object identity and one-hot context indicator
- **Output**: Context-independent features (apply across all contexts) plus context-specific features (apply in single context only)

This setup explicitly requires:
1. Feature learning (weights must change to learn structure)
2. Nonlinearity (linear networks cannot solve it)
3. Weight sharing (no separate network per context)

### Learning Dynamics

Under neural race reduction assumptions, the dynamics decompose into independent 1D networks. Each singular mode evolves as:

$$\pi_\alpha(t) = \frac{S_\alpha/D_\alpha}{1 - (1 - S_\alpha/(D_\alpha \cdot \pi_0)) \cdot \exp(-2S_\alpha \cdot t/\tau)}$$

where $S_\alpha$ and $D_\alpha$ are singular values of input-output and input correlation matrices respectively.

**Critical insight**: Pathway effective datasets are path-dependent. Each linear pathway through the network observes different data statistics depending on which samples activate its gating pattern.

---

## Key Results

### Mixed Selectivity vs Modularity

The winning strategy exhibits **structured mixed selectivity**:
- Pathways are active in multiple contexts (not one context each)
- Each pathway learns context-specific residuals despite shared activation
- Collectively, pathways produce correct outputs through interference patterns

**Concrete example** (3 contexts):
- One "common" pathway receives all contexts
- Three context-specific pathways are each active for 2 of 3 contexts
- NOT: Three modular pathways each dedicated to one context

### Why Mixed Selectivity Wins

**Learning speed optimization**: Pathways that observe more data (larger effective dataset) learn faster (higher singular values $S_\alpha$). Mixed selectivity increases $S_\alpha$ by enlarging the effective dataset each pathway observes.

The mathematical mechanism: In Lotka-Volterra dynamics, convergence rate is $\propto S_\alpha$. Pathways with larger $S_\alpha$ win the neural race. Mixed selectivity is the winning strategy because it:
1. Maximizes data utilization per pathway
2. Still maintains computational differentiation across contexts
3. Produces lower loss trajectory than modular alternatives

### Effect of More Contexts

As contexts increase from $C=3$ to $C=4$ to $C=5$:
- Mixed selectivity intensifies
- Context-specific pathways remain active for $(C-1)$ contexts (not just one)
- Singular value dynamics scale as $(C-1)S_\alpha/D_\alpha$

### Effect of Depth

Adding a second ReLU layer:
- Removes constraints from Lemma 4.1
- First layer becomes completely shared (no nonlinearity applied)
- Nonlinearity and gating concentrated in deeper layers
- **Mixed selectivity persists** as a fundamental learning strategy, not merely a constraint solution

---

## Answers to First-pass Questions

### 1. How does the theory extend from GDLNs to standard ReLU networks?

Through the ReLN construction: For any ReLU network trajectory under training, there exists a unique GDLN with matching gates at each timestep. This GDLN is analytically tractable while being equivalent to the ReLU network. The key assumption is that gates stabilize sufficiently for the neural race reduction to apply.

### 2. What specific mathematical conditions lead to structured mixed selectivity vs strictly modular representations?

**Modular representations** would require:
- Pathways active in only one context each
- No sharing of hidden neurons across contexts

**Mixed selectivity emerges when**:
- Learning speed matters (gradient descent favors faster convergence)
- Pathways that see more data learn faster
- Sharing neurons across contexts increases effective dataset size
- The task can be solved through interference patterns rather than strict separation

The key condition is the singular value structure: mixed-selective pathways have higher $S_\alpha$ and therefore win the neural race.

### 3. How does depth affect structured mixed selectivity?

Deeper networks show:
- More flexibility in gating pattern allocation
- Tendency to push nonlinearity to later layers
- Earlier layers become shared (linear) across contexts
- Mixed selectivity becomes a design principle, not just a constraint solution

### 4. Can these insights be validated empirically?

Yes, the paper includes experiments showing:
- 2-layer ReLU networks develop predicted mixed selectivity patterns
- Deeper networks show increased sharing in early layers
- More contexts increase mixed selectivity intensity
- Learning speed correlates with pathway sharing

### 5. Practical implications for architecture design?

- Don't force modularity; let networks find structured sharing
- Expect mixed selectivity to emerge naturally in multi-task settings
- Depth enables more flexible sharing strategies
- Learning speed bias can be leveraged by designing tasks that encourage sharing

### 6. Connection to biological mixed selectivity?

The paper provides a normative account: mixed selectivity in biological neurons may emerge from similar learning speed pressures. This connects to neuroscience observations that neurons are often active for multiple stimuli/contexts rather than being strictly selective.

---

## Critical Analysis for Neural Race Mismatch

### Key Insight: Single-Task vs Multi-Task Distinction

The Jarvis paper studies **multi-task contextual learning** where:
- Different contexts require different input-output mappings
- Shared representations must handle multiple tasks
- Competition is between different ways to solve multiple tasks

Our experiments use **single-task multi-view classification** where:
- All views map to the same label
- There's only one task (classification)
- No multi-task competition pressure

**This is a critical difference.** The neural race dynamics in Saxe et al. 2022 and Jarvis et al. 2025 both assume multi-task structure that creates competition between pathways serving different tasks.

### Why Our Setup Doesn't Show Winner-Take-All

1. **No task-based competition**: In single-task classification, all views serve the same goal. There's no reason for pathways to compete.

2. **Orthogonal views don't compete**: Our views occupy different input slots without overlap. The network can simply learn all views in parallel.

3. **No shared output constraint**: Multi-task setups force pathways to share output capacity. Our single output head has no such constraint.

4. **Mixed selectivity doesn't apply**: Mixed selectivity is about sharing neurons across contexts/tasks. With one task, there's nothing to share across.

### The "Make Haste Slowly" Insight

The paper's title refers to the paradox: ReLU networks perform **slow feature learning** (weights meaningfully change) by exploiting **fast learning speed** (node reuse and data efficiency). This creates an implicit bias toward mixed selectivity.

In our setup:
- There's no speed advantage to reusing nodes across views
- Each view provides independent gradient signal
- The network can "make haste quickly" by learning all views in parallel

### Implications for Understanding the Mismatch

The Jarvis paper confirms that **neural race dynamics depend on multi-task structure**:

1. **Competition emerges from task structure**, not just capacity constraints
2. **Single-task setups lack the competition pressure** that drives winner-take-all
3. **Our identical-seed results make sense**: Without competition, there's no race, so no stochasticity

This explains our finding that 10 different seeds produce identical results (dominance = 0.379 +/- 0.000). There is no race because there is no multi-task competition.

---

## Relevance to Lab Vision

### Direct Relevance to neural-race-multiview

This paper provides **critical theoretical insight** into our theory-experiment mismatch:

1. **Confirms task structure matters**: Winner-take-all requires multi-task competition, not just multi-view data
2. **Explains why our experiments fail to show competition**: Single-task classification lacks the structural preconditions
3. **Suggests a path forward**: To see neural race dynamics, we need multi-task structure (different views -> different labels or different tasks)

### Strategic Implications

**Option A (revise theory)**: The paper supports reframing our contribution. We're not studying "neural race in multi-view classification" but rather "conditions for pathway competition."

**Option B (find emergent competition)**: The paper suggests multi-task setups as the place to look. If views predict different auxiliary tasks, competition might emerge.

**Option C (reframe contribution)**: The paper's mixed selectivity findings could inform a new angle: "Why do multi-view networks learn everything? Because single-task structure lacks competition pressure."

---

## Potential Connections

- **Connection to Saxe 2022**: Same theoretical framework, extended to finite-width ReLU. Confirms and extends neural race reduction.
- **Connection to multi-task learning literature**: Mixed selectivity relates to positive transfer and interference in multi-task optimization.
- **Gap revealed**: Need to distinguish "multi-view" (different input representations of same concept) from "multi-task" (different objectives) when applying neural race theory.
- **Neuroscience connection**: Provides normative account for biological mixed selectivity that could inform interpretability research.

---

## Ideas Sparked

- **IDEA-002 (update)**: Test neural race dynamics with multi-task structure: View 1 -> Task A, View 2 -> Task B. This should show competition per the theory.

- **IDEA-003**: Study the transition from "learn everything" (single-task) to "winner-take-all" (multi-task) as we gradually introduce task conflict.

- **IDEA-004**: Investigate whether knowledge distillation's effectiveness relates to mixed selectivity. If teacher exhibits mixed selectivity, does student inherit it?

---

## Summary: What This Means for Our Project

### The Core Insight

**The neural race reduction applies to multi-task learning, not single-task multi-view classification.**

Our experiments show:
- No winner-take-all dynamics
- All seeds produce identical results
- All views are learned

This is **not a failure of the theory** but rather **an application to the wrong problem class**.

### Recommended Path Forward

1. **Acknowledge the task structure distinction** in paper framing
2. **Either** pivot to multi-task experiments that should show neural race dynamics
3. **Or** reframe contribution around "conditions for pathway competition" with single-task as the non-competitive baseline
4. **Consider** mixed selectivity as a phenomenon to study: When does it emerge? How does KD affect it?

### Key Takeaway for Advisor Discussion

The Jarvis paper provides theoretical clarity: winner-take-all emerges from multi-task competition pressure, which our single-task setup lacks. This suggests our experiments are correct but our theoretical framing needs revision. The question shifts from "why doesn't neural race happen?" to "what problem structures induce neural race?"

---

## Technical Details (for future reference)

### The Lotka-Volterra Connection

Both papers frame dynamics as competitive Lotka-Volterra systems:
$$\frac{ds_m}{dt} = r_m s_m (1 - \sum_{m'} s_{m'}^2 / K_{cap})$$

where:
- $r_m = \sigma_1(\Sigma_{y,m})$ is growth rate (from correlation strength)
- $K_{cap} = s_{max}^2$ is shared carrying capacity

Mixed selectivity increases $r_m$ by increasing effective data per pathway.

### The ReLN Construction

Key definitions:
- **GDLN**: Network with explicit gating matrices $D_\ell(x)$
- **ReLN**: GDLN with gates matching ReLU behavior at all timesteps
- **Neural race winning strategy**: The gating pattern that achieves fastest loss reduction

Proposition 4.2 establishes that the ReLN solving the neural race gives the optimal ReLU network behavior.

### Depth Analysis

For 2-layer networks:
- Layer 1 gating determines pathway structure
- Deeper layers enable richer interference patterns
- Mixed selectivity persists and intensifies with depth

---

*Summarized by: Reading Stack Agent*
*Last updated: 2026-01-13*
*Source paper: arXiv:2503.06181*
