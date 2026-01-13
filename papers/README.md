# Papers

Each paper is a separate git repository linked as a submodule.

## Active Papers

- [neural-race-multiview](neural-race-multiview/) - Unifying Neural Race Reduction with Multi-View Theory (NeurIPS 2026)

## Working with Paper Submodules

### Clone ai-lab with all papers
```bash
git clone --recurse-submodules https://github.com/YOURORG/ai-lab.git
```

### Add a new paper
```bash
# Create paper repo separately
# Then add as submodule:
git submodule add https://github.com/YOURORG/new-paper.git papers/new-paper
git commit -m "Add new-paper as submodule"
```

### Update a paper
```bash
cd papers/neural-race-multiview
git add .
git commit -m "Your changes"
git push

cd ../..
git add papers/neural-race-multiview
git commit -m "Update neural-race-multiview to latest"
git push
```

### Pull latest papers
```bash
git pull
git submodule update --remote --merge
```

### Remove a paper
```bash
git submodule deinit papers/old-paper
git rm papers/old-paper
git commit -m "Remove old-paper"
```

## Creating a New Paper

When adding new papers:

1. Create paper directory: `papers/new-paper/`
2. Initialize as git repo: `cd papers/new-paper && git init`
3. Add content and commit
4. Create remote repo on GitHub
5. Add remote: `git remote add origin <url>`
6. Push: `git push -u origin main`
7. Add as submodule to main repo: `cd ../.. && git submodule add <url> papers/new-paper`
8. Commit submodule: `git commit -m "Add new-paper as submodule"`

## Paper Structure

Each paper should follow the ai-lab structure:

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

## Benefits of Submodules

- **Clean publication**: Each paper repo is self-contained and citable
- **Independent history**: Each paper has its own git log, tags, releases
- **Paper lifecycle**: Archive, delete, or privatize papers independently
- **Flexible access**: Different collaborators for different papers
- **Unified workflow**: Work in single directory structure locally
