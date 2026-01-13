# Reading Stack

A system for managing papers to read, generating summaries, and developing research ideas.

## Purpose

- **Inbox**: Queue of papers to read (arXiv links)
- **Summaries**: Processed papers with focus area tags
- **Ideas**: Structured pipeline from spark to paper PRD

## Directory Structure

```
reading_stack/
├── inbox/           # Papers waiting to be read
├── summaries/       # Processed paper summaries
└── ideas/
    ├── nascent/     # Raw ideas from reading
    ├── developing/  # Ideas being refined
    └── ready/       # Ready for paper PRD
```

## Workflow

```
User drops arXiv link → inbox/PAPER-XXX.md
         ↓
Read and summarize → summaries/PAPER-XXX.md (with focus tags)
         ↓
Ideas sparked → ideas/nascent/IDEA-XXX.md
         ↓
Refine idea → ideas/developing/IDEA-XXX.md
         ↓
Ready for paper → ideas/ready/IDEA-XXX.md
         ↓
Create paper PRD → papers/<name>/prd/
```

---

## Paper Entry Format (inbox/)

Filename: `PAPER-XXX-[first-author]-[year].md`

```markdown
# Paper: [Title]

- **ID**: PAPER-XXX
- **arXiv**: 2401.12345
- **Authors**: Smith et al.
- **Year**: 2024
- **Added**: YYYY-MM-DD
- **Status**: unread

## Why Read
Brief note on why this paper is relevant to lab vision.

## Focus Areas
- [ ] Mechanistic DL Theory
- [ ] Feature Learning
- [ ] Knowledge Distillation
- [ ] Theory-Inspired Applications
```

---

## Summary Format (summaries/)

Filename: `PAPER-XXX-[first-author]-[year].md`

```markdown
# Summary: [Title]

- **Paper ID**: PAPER-XXX
- **arXiv**: 2401.12345
- **Summarized**: YYYY-MM-DD

## Focus Area Tags
- Mechanistic DL Theory
- Feature Learning

## One-Line Summary
[Single sentence capturing the core contribution]

## Key Contributions
1. ...
2. ...

## Methodology
[Brief description of approach]

## Key Results
- ...

## Relevance to Lab Vision
[How this connects to our research direction]

## Potential Connections
- Connection to [other paper/idea]
- Gap this reveals: ...

## Ideas Sparked
- IDEA-XXX: [brief description]
```

---

## Idea Pipeline

### Stage 1: Nascent (ideas/nascent/)

Raw sparks from reading. Filename: `IDEA-XXX-[working-title].md`

```markdown
# Idea: [Working Title]

- **ID**: IDEA-XXX
- **Stage**: nascent
- **Created**: YYYY-MM-DD
- **Source Papers**: PAPER-001, PAPER-002

## Spark
[What's the core insight or question?]

## Focus Areas
- Feature Learning
- Knowledge Distillation

## Initial Thoughts
[Rough notes, could be messy]
```

### Stage 2: Developing (ideas/developing/)

Ideas being refined. Filename: `IDEA-XXX-[working-title].md`

```markdown
# Idea: [Working Title]

- **ID**: IDEA-XXX
- **Stage**: developing
- **Created**: YYYY-MM-DD
- **Promoted**: YYYY-MM-DD
- **Source Papers**: PAPER-001, PAPER-002

## Research Question
[Clear articulation of the question]

## Hypothesis
[Testable prediction]

## Why Novel
[How this differs from existing work]

## Potential Experiments
- ...

## Open Questions
- ...

## Promotion Criteria
- [ ] Clear research question
- [ ] Testable hypothesis
- [ ] Novelty argument
- [ ] Viable experiment plan
```

### Stage 3: Ready (ideas/ready/)

Ready for paper PRD. Filename: `IDEA-XXX-[working-title].md`

```markdown
# Idea: [Working Title]

- **ID**: IDEA-XXX
- **Stage**: ready
- **Created**: YYYY-MM-DD
- **Ready**: YYYY-MM-DD
- **Source Papers**: PAPER-001, PAPER-002

## Research Question
...

## Hypothesis
...

## Why Novel
...

## Proposed Experiments
...

## Target Venue
[Conference/journal and deadline]

## Next Step
Create paper PRD in papers/<paper-name>/prd/
```

---

## Promotion Criteria

### nascent → developing

- Has clear research question (not just "interesting observation")
- Connects to at least one lab_vision focus area
- Not obviously scooped or incremental

### developing → ready

- [ ] Clear, specific research question
- [ ] Testable hypothesis with predicted outcome
- [ ] Novelty argument (why this isn't already done)
- [ ] Viable experiment plan (low compute, provable on small data)
- [ ] Aligns with lab_vision strategic priorities

### ready → paper PRD

1. Create new paper directory in `papers/`
2. Move idea content to `papers/<name>/prd/paper_requirements.md`
3. Log handoff in `agents/context/handoffs.md`

---

## Focus Areas (from lab_vision.md)

When tagging papers and ideas, use these focus areas:

1. **Mechanistic DL Theory** - How/why neural networks learn, learning dynamics
2. **Feature Learning** - How representations emerge, what determines features learned
3. **Knowledge Distillation** - Theoretical foundations, student-teacher dynamics
4. **Theory-Inspired Applications** - Applied methods from theoretical understanding

---

## ID Conventions

- Papers: `PAPER-001`, `PAPER-002`, etc.
- Ideas: `IDEA-001`, `IDEA-002`, etc.

IDs are assigned sequentially across all stages. An idea keeps its ID as it moves through the pipeline.
