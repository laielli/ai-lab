# Paper: From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence

- **ID**: PAPER-004
- **arXiv**: 2601.03220
- **Authors**: Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, Andrew Gordon Wilson
- **Year**: 2026
- **Added**: 2026-01-13
- **Status**: read
- **Summary**: reading_stack/summaries/PAPER-004-finzi-2026.md

## Why Read

This paper introduces "epiplexity," a new information-theoretic framework that addresses fundamental limitations of classical information theory (Shannon entropy, Kolmogorov complexity) by accounting for computational constraints. The work is highly relevant to mechanistic DL theory as it provides a principled way to understand what computationally bounded learners can extract from data, potentially offering new theoretical tools for understanding neural network learning dynamics and feature emergence.

## Focus Areas
- [x] Mechanistic DL Theory
- [x] Feature Learning
- [ ] Knowledge Distillation
- [x] Theory-Inspired Applications

## Notes

Key concepts from abstract:
- Challenges three classical assumptions: (1) information cannot be increased by deterministic transformations, (2) information is independent of data order, (3) likelihood modeling is merely distribution matching
- Proposes epiplexity to quantify learnable content for computationally constrained observers
- Shows that computation can create information and data ordering matters
- Provides practical estimation methods
- Demonstrates correlation with downstream task performance and improved generalization
- Applications to data selection (understudied complement to model selection)

Subject classifications: cs.LG, stat.ML

## Questions

1. How does epiplexity differ technically from Kolmogorov complexity and Shannon entropy? What are the formal definitions?
2. What is the computational model used for the "bounded observer"? Is it related to specific neural network architectures?
3. How does the practical estimation method work? Is it tractable for realistic datasets?
4. What are the specific mechanisms by which computation "creates" information in this framework?
5. How does data ordering affect epiplexity? What are the implications for curriculum learning or data augmentation?
6. Can epiplexity provide insights into why certain architectures learn certain features (feature learning connection)?
7. What are the implications for understanding learning dynamics in neural networks?
8. How does this relate to existing work on data complexity measures or learning theory bounds?
