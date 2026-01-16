# Papers

Each paper is a separate git repository, independent from the main ai-lab repo.

The `papers/` directory is gitignored — the main repo does not track paper contents or versions.

## Active Papers

- [neural-race-multiview](neural-race-multiview/) - Unifying Neural Race Reduction with Multi-View Theory (NeurIPS 2026)

## Working with Papers

### Clone a paper
```bash
cd papers
git clone https://github.com/YOURORG/paper-name.git
```

### Update a paper
```bash
cd papers/<paper_name>
git add .
git commit -m "Your changes"
git push
```

No changes to the main repo needed.

### Pull latest changes
```bash
cd papers/<paper_name>
git pull
```

## Creating a New Paper

1. Create paper directory: `mkdir papers/new-paper && cd papers/new-paper`
2. Initialize git repo: `git init`
3. Add content and commit
4. Create remote repo on GitHub
5. Add remote: `git remote add origin <url>`
6. Push: `git push -u origin main`

## Paper Structure

Each paper should follow this structure:

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

## Benefits of Independent Repos

- **Clean publication**: Each paper repo is self-contained and citable
- **Independent history**: Each paper has its own git log, tags, releases
- **Simple workflow**: No submodule pointer updates needed
- **Paper lifecycle**: Archive, delete, or privatize papers independently
- **Flexible access**: Different collaborators for different papers
