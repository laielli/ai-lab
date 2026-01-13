# Paper: Make Haste Slowly: A Theory of Emergent Structured Mixed Selectivity in Feature Learning ReLU Networks

- **ID**: PAPER-002
- **arXiv**: 2503.06181
- **Authors**: Devon Jarvis, Richard Klein, Benjamin Rosman, Andrew M. Saxe
- **Year**: 2025
- **Added**: 2026-01-13
- **Status**: summarized

## Why Read
This paper addresses fundamental gaps in understanding feature learning in finite-dimensional ReLU networks by connecting them to Gated Deep Linear Networks. It's highly relevant to the lab's focus on mechanistic DL theory and feature learning, particularly the emergence of structured representations. The finding that ReLU networks develop "structured mixed selectivity" rather than strictly modular representations could inform our understanding of how neural networks balance reusability and task-specific adaptation. Accepted at ICLR 2025.

## Focus Areas
- [x] Mechanistic DL Theory
- [x] Feature Learning
- [ ] Knowledge Distillation
- [ ] Theory-Inspired Applications

## Notes
First author: Devon Jarvis
Venue: ICLR 2025 (accepted)
Key insight: ReLU networks develop an inductive bias toward structured mixed-selectivity representations that emerge from node-reuse pressures and learning speed optimization.
Connection to Gated Deep Linear Networks provides theoretical framework.
Multi-task learning context requiring both feature learning and nonlinearity.

## Questions
- How does the theory extend from Gated Deep Linear Networks to standard ReLU networks? What are the key differences and assumptions?
- What specific mathematical conditions lead to structured mixed selectivity versus strictly modular representations?
- How does the depth of the network affect the emergence of structured mixed selectivity?
- Can these theoretical insights be validated empirically on standard benchmarks?
- What are the practical implications for architecture design and multi-task learning systems?
- How does this relate to biological neural selectivity patterns in the brain?
