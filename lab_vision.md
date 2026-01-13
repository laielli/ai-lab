# Lab Vision

## Mission

Develop mechanistic understanding of deep learning through theory-experiment cycles. Bridge theoretical frameworks with empirical validation. Make predictions that change how people think about neural networks.

## Research Identity

### Core Approach

- **Theory-driven**: Start with theoretical questions, design experiments to test
- **Mechanistic**: Explain WHY things work, not just THAT they work
- **Unifying**: Connect disparate frameworks and literatures
- **Predictive**: Theories must make testable quantitative predictions

### Values

- **Depth over breadth**: Deep understanding of few problems > shallow coverage
- **Rigor over speed**: Correct understanding > fast publication
- **Insight over performance**: Explaining mechanisms > achieving SOTA

## Focus Areas (In Scope)

### Primary Themes

1. **Mechanistic Deep Learning Theory**
   - How and why neural networks learn
   - Learning dynamics and their consequences
   - Theoretical frameworks for understanding training

2. **Feature Learning**
   - How representations emerge during training
   - What determines which features are learned
   - Connections between architecture, data, and learned features

3. **Knowledge Distillation**
   - Theoretical foundations of knowledge transfer
   - Why and when distillation works
   - Mechanistic accounts of student-teacher dynamics

4. **Theory-Inspired Applications**
   - Applied methods that emerge from theoretical understanding
   - Practical techniques motivated by mechanistic insight
   - Design principles derived from theory

### Subdomain Keywords (for search/exploration)

neural race reduction, multi-view learning, lottery tickets, feature emergence, gradient dynamics, loss landscape, representation learning, teacher-student, dark knowledge, soft labels, capacity utilization

## Research Taste

### Good Problems (pursue)

- Fills genuine gaps in theoretical understanding
- Makes quantitative predictions that can be validated empirically
- Connects previously separate theoretical frameworks
- Reveals mechanisms that explain observed phenomena
- Has implications beyond the specific setting

### Bad Problems (avoid)

- Pure engineering without theoretical insight
- Incremental improvements to existing methods
- Requires massive compute without conceptual contribution
- Phenomenological description without mechanistic explanation
- Narrow applicability with no broader implications

### Paper Quality Signals

- Changes how readers think about the problem
- Predictions are testable AND tested
- Unifies previously separate understandings
- Opens new research directions

### Strategic Priorities

- **Unique and creative over incremental**: Prioritize contributions that do not linearly follow from recent papers. Non-obvious directions reduce scoop risk and increase long-term field impact.
- **Connect disparate subfields**: Seek ideas that bridge separate research communities. Cross-pollination increases novelty and reduces competition from researchers siloed in single areas.
- **Low compute requirements**: Prioritize ideas provable on small datasets or toy settings. Concepts should not require scaling laws or massive runs to validate. This democratizes verification and speeds iteration.

## Non-Goals (Out of Scope)

Explicit boundaries:

- Pure benchmarking papers (performance tables without insight)
- Large-scale training without theoretical motivation
- Method papers without mechanistic understanding
- Survey/review papers (we create new knowledge)

## Agent Instructions

### Explorer Agent

When searching literature or identifying research directions:

- Prioritize papers with mechanistic explanations
- Look for connections between Focus Areas
- Flag theoretical frameworks that could be unified
- Search using Subdomain Keywords above

### Paper Agent

When evaluating paper proposals:

- Check alignment with Focus Areas
- Verify Research Taste criteria are met
- Ensure mechanistic contribution is clear
- Reject if it falls under Non-Goals

### Orchestrator

When prioritizing work:

- Advance papers that strongly align with Focus Areas
- Deprioritize work drifting toward Non-Goals
- Escalate scope questions to lab vision review
