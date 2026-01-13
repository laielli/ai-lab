# Task Queue

Tasks flow through: `backlog/` → `active/` → `review/` → done

## Task File Template

```markdown
# Task: <title>

- **ID**: TASK-XXX
- **Owner**: <agent-name>
- **Paper**: <paper-name>
- **Created**: YYYY-MM-DD

## Description

<what needs to be done>

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Handoff Notes

<context from previous agent>
```

## Workflow

1. **Backlog** — Tasks not yet started
2. **Active** — Currently being worked on
3. **Review** — Awaiting approval or handoff
4. **Done** — Archive or delete completed tasks

## Creating a Task

1. Create markdown file in `backlog/` with unique ID
2. Fill in template with clear acceptance criteria
3. Move to `active/` when starting work
4. Move to `review/` when ready for approval
5. Archive or delete when complete
