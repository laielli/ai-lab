# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an AI Lab framework designed for agent-driven research where the goal is publishing papers at top-tier AI conferences (e.g., NeurIPS). The framework treats "Papers" as equivalent to "Products" and uses Paper Requirements Documents (PRDs) instead of traditional product requirements.

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
