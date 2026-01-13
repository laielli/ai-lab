# AI Lab Framework

A simplified directory structure for AI-agent-driven labs where the priority is to publish research papers at top-tier AI conferences (e.g. NeurIPS): "Papers" are equivalent to "Products"; PRD = "Paper Requirements Document". Optimized for clarity, minimal maintenance, and effective agent collaboration.

## Directory Structure

```
<company>/
├── README.md
├── CLAUDE.md                    # Instructions for AI assistants
├── AGENTS.md                    # How agents collaborate
│
├── standards/                   # Constraints all agents follow
│   ├── engineering.md           # Coding conventions, reviews, CI
│   ├── ml.md                    # Eval discipline, dataset versioning
│   ├── paper.md                 # PRD template, success metrics
│   ├── writing.md               # Paper writing formats, and styles
│   ├── search.md                # Literature search best practices
│   └── decisions.md             # ADR-style decision log
│
├── shared_stack/                # Common technical assets
│   ├── ml/                      # Models, training, evaluation
│   └── datasets.md              # ADR-style decision log
│
├── papers/                      # Paper-specific work
│   └── <paper_name>/
│       ├── README.md
│       ├── prd/                 # Paper requirements
│       ├── specs/               # Technical specifications
│       ├── log/                 # Experiment log
│       ├── src/                 # Experiment source code
│       ├── evals/               # Paper-specific evaluations
│       ├── paper/               # Paper drafts in latex
│       ├── presentation/        # Presentation materials
│       └── datasets/            # UX flows, copy, assets
│
├── execution/                   # Experiment operations
│   ├── roadmap.md               # Now / next / later priorities
│   ├── submissions.md           # Submission process and calendar
│   └── metrics.md               # North star metrics
│
└── agents/                      # Agent orchestration
    ├── charters/                # Agent definitions (one per type)
    │   ├── orchestrator.md      # Priority calls, routing, blockers
    │   ├── explorer.md          # Literature search, novelty discovery, disparate connections
    │   ├── paper.md             # PRDs, specs, acceptance criteria
    │   ├── engineering.md       # Implementation, review, quality
    │   ├── ml.md                # Models, evals, training
    │   └── communication.md     # Explanation, presentation, content
    │
    ├── tasks/                   # Work queue
    │   ├── backlog/             # Not yet started
    │   ├── active/              # In progress
    │   └── review/              # Awaiting approval
    │
    └── context/                 # Shared state
        ├── priorities.md        # Current focus areas
        ├── blockers.md          # Known issues
        └── handoffs.md          # Cross-agent notes
```

---

## How It Works

### Agents by Function

Instead of mirroring human org charts, define agents by what they do:

| Agent | Responsibility |
|-------|----------------|
| **orchestrator** | Prioritization, routing, unblocking |
| **explorer** | Literature search, novelty discovery, disparate connections |
| **paper** | Requirements, specs, acceptance criteria |
| **engineering** | Implementation, code review, quality |
| **ml** | Models, training, evaluation |
| **communication** | Explanation, presentation, content |

### Task Flow

Work moves through a simple state machine:

```
backlog/ → active/ → review/ → (done or back to backlog)
```

Each task is a markdown file with:
- Owner (which agent)
- Paper (which paper it's for)
- Status (current state)
- Description and acceptance criteria
- Handoff notes from previous agent

### Shared Context

All agents read from `agents/context/` for:
- **priorities.md** — What matters now
- **blockers.md** — What's stuck
- **handoffs.md** — Recent cross-agent communications

---

## Task File Format

```markdown
# Task: <title>

- **ID**: TASK-001
- **Owner**: engineering
- **Paper**: kdmech
- **Created**: 2025-01-11

## Description

<what needs to be done>

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Handoff Notes

<context from previous agent>
```

---

## Charter File Format

```markdown
# Agent: <name>

## Scope

What this agent is responsible for.

## Inputs

- Where it gets work
- What context it needs

## Outputs

- What it produces
- Where results go

## Constraints

- What it should NOT do
- When to escalate
```

---

## Design Principles

1. **Flat over nested** — Avoid deep directory hierarchies
2. **Explicit over implicit** — Charters define scope, not folder structure
3. **Centralized routing** — One task queue, not bilateral exchanges
4. **Products are separate** — Agent structure doesn't change when products are added
5. **Minimal ceremony** — Add structure only when needed
