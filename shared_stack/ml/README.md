# Shared ML Stack

Common ML utilities, models, and training code used across papers.

## Structure

```
ml/
├── models/          # Shared model architectures
├── training/        # Training utilities and scripts
├── evaluation/      # Evaluation metrics and tools
└── preprocessing/   # Data preprocessing utilities
```

## Usage

Add reusable ML components here that are used by multiple papers. Paper-specific code should live in `papers/<paper_name>/src/`.
