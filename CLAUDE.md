# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an AI Lab framework designed for agent-driven research where the goal is publishing papers at top-tier AI conferences (e.g., NeurIPS). The framework treats "Papers" as equivalent to "Products" and uses Paper Requirements Documents (PRDs) instead of traditional product requirements.

## Research Direction

For lab research focus, scope, and taste, see `lab_vision.md`. All agents should consult this document when making decisions about research direction, evaluating paper proposals, or prioritizing work.

## Reading Stack

For paper reading queue, summaries, and idea pipeline, see `reading_stack/README.md`. Papers flow through: inbox → summaries → ideas (nascent → developing → ready) → paper PRD.

## Code Stack

For code repository review queue and implementation analysis, see `code_stack/README.md`. Repos flow through: inbox → summaries (with quality assessment and key implementation details). Cross-references with reading_stack when repos correspond to papers.

## Directory Structure Philosophy

The repository follows these design principles:

1. **Flat over nested** — Avoid deep directory hierarchies
2. **Explicit over implicit** — Charters define scope, not folder structure
3. **Centralized routing** — One task queue, not bilateral exchanges
4. **Papers are separate** — Agent structure doesn't change when papers are added
5. **Minimal ceremony** — Add structure only when needed

## Key Directory Layout

```
ai-lab/
├── standards/          # Constraints all agents follow (engineering, ml, paper, writing, search, decisions)
├── shared_stack/       # Common technical assets (ml models, training, datasets)
├── reading_stack/      # Paper reading queue, summaries, and idea pipeline
├── code_stack/         # Code repo review queue and implementation analysis
├── papers/             # Paper-specific work (one subdirectory per paper)
├── execution/          # Operations (roadmap, submissions, metrics)
└── agents/             # Agent orchestration (charters, tasks, context)
```

## Agent System

Work is organized by function-based agents, not human org charts:

- **orchestrator** — Prioritization, routing, unblocking
- **explorer** — Literature search, novelty discovery, disparate connections
- **paper** — Requirements, specs, acceptance criteria
- **engineering** — Implementation, code review, quality
- **ml** — Models, training, evaluation
- **communication** — Explanation, presentation, content

## Task Management

Tasks flow through: `backlog/` → `active/` → `review/` → done

Task files should be markdown with:
- ID, Owner (agent), Paper, Created date
- Description and acceptance criteria
- Handoff notes from previous agent

Example task file location: `agents/tasks/active/TASK-001.md`

## Shared Context

All agents coordinate through `agents/context/`:
- `priorities.md` — Current focus areas
- `blockers.md` — Known issues preventing progress
- `handoffs.md` — Recent cross-agent communications

## Paper Development Workflow

Each paper lives in `papers/<paper_name>/` with:
- `prd/` — Paper requirements and objectives
- `specs/` — Technical specifications
- `log/` — Experiment log and results
- `src/` — Experiment source code
- `evals/` — Paper-specific evaluations
- `paper/` — Paper drafts in LaTeX
- `presentation/` — Presentation materials
- `datasets/` — Paper-specific datasets

## Standards to Follow

Before working on specific areas, consult the relevant standards documents:
- `standards/engineering.md` — Coding conventions, reviews, CI
- `standards/ml.md` — Evaluation discipline, dataset versioning
- `standards/paper.md` — PRD template, success metrics
- `standards/writing.md` — Paper writing formats and styles
- `standards/search.md` — Literature search best practices
- `standards/decisions.md` — ADR-style decision log

## When Creating New Work

1. Check `agents/context/priorities.md` for current focus
2. Review relevant agent charter in `agents/charters/`
3. Create task file in appropriate queue (`backlog/`, `active/`, or `review/`)
4. Follow the task file format with clear acceptance criteria
5. Update shared context files when blocking issues arise

## Git Workflow

### Repository Structure
- **Main repo** (ai-lab): Lab infrastructure, standards, agents, execution
- **Paper repos**: Independent git repositories in `papers/` directory (not tracked by main repo)

The `papers/` directory is in `.gitignore`. Each paper is its own git repository, fully independent from the main repo.

### Working on Lab Infrastructure

When modifying files in the main repo (standards, agents, execution, shared_stack):

```bash
git add <files>
git commit -m "Description of changes"
git push
```

### Working on a Paper

Papers have their own git repositories. Work directly in the paper directory:

```bash
cd papers/<paper_name>
git add .
git commit -m "Description of changes"
git push
```

No changes to the main repo are needed — papers are fully independent.

### Adding a New Paper

To add a new paper, create or clone a git repository in the `papers/` directory:

```bash
cd papers
git clone <paper-repo-url>
# or: mkdir <paper_name> && cd <paper_name> && git init
```

The main repo will not track it (papers/ is gitignored).
