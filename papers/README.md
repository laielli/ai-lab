# Papers

Each paper gets its own subdirectory with the following structure:

```
<paper_name>/
├── README.md           # Paper overview
├── prd/                # Paper requirements document
├── specs/              # Technical specifications
├── log/                # Experiment log
├── src/                # Experiment source code
├── evals/              # Paper-specific evaluations
├── paper/              # Paper drafts (LaTeX)
├── presentation/       # Presentation materials
└── datasets/           # Paper-specific datasets
```

## Creating a New Paper

1. Create directory: `mkdir papers/<paper_name>`
2. Copy the structure above
3. Create a PRD following `standards/paper.md`
4. Update `execution/roadmap.md` with paper milestones
