# Code Stack

A system for reviewing code repositories that correspond to research papers, understanding implementations, and assessing code quality and utility.

## Purpose

- **Inbox**: Queue of repos to review (GitHub links)
- **Summaries**: Processed repo assessments with implementation notes

## Directory Structure

```
code_stack/
├── inbox/           # Repos waiting to be reviewed
└── summaries/       # Processed repo summaries
```

## Workflow

```
User drops GitHub link → inbox/REPO-XXX.md
         ↓
Review codebase → summaries/REPO-XXX.md (with assessments)
         ↓
Link to paper summary → reading_stack/summaries/PAPER-XXX.md (cross-reference)
```

---

## Repo Entry Format (inbox/)

Filename: `REPO-XXX-[repo-name].md`

```markdown
# Repo: [Repo Name]

- **ID**: REPO-XXX
- **GitHub**: https://github.com/owner/repo
- **Paper**: PAPER-XXX (if applicable)
- **Added**: YYYY-MM-DD
- **Status**: unread

## Why Review
Brief note on why this repo is relevant to lab research.

## Key Methods to Understand
- [ ] Method 1 (e.g., "Neural Race Reduction dynamics")
- [ ] Method 2 (e.g., "View selection mechanism")

## Initial Notes
Notes from initial browsing (README, structure, etc.)
```

---

## Summary Format (summaries/)

Filename: `REPO-XXX-[repo-name].md`

```markdown
# Summary: [Repo Name]

- **Repo ID**: REPO-XXX
- **GitHub**: https://github.com/owner/repo
- **Paper**: PAPER-XXX
- **Reviewed**: YYYY-MM-DD

## One-Line Summary
[Single sentence capturing what this repo does]

## Codebase Overview

### Structure
[High-level directory structure and organization]

### Tech Stack
- Language: Python 3.x
- Framework: PyTorch / JAX / etc.
- Key dependencies: ...

### Size & Complexity
- Lines of code: ~X,XXX
- Core modules: X
- Complexity assessment: [simple | moderate | complex]

## Quality Assessment

### Code Quality
- **Readability**: [poor | fair | good | excellent]
- **Documentation**: [none | minimal | adequate | thorough]
- **Tests**: [none | minimal | adequate | comprehensive]
- **Reproducibility**: [difficult | possible | easy]

### Utility Assessment
- **Reusable components**: [list any]
- **Adaptation difficulty**: [trivial | moderate | significant | impractical]
- **Active maintenance**: [abandoned | sporadic | active]

## Key Implementation Details

### [Method 1 Name]
**Location**: `path/to/file.py:L100-200`

**Summary**: [Brief description of how this method is implemented]

**Key Insights**:
- ...
- ...

**Differences from Paper**: [If any]

### [Method 2 Name]
**Location**: `path/to/file.py:L300-400`

**Summary**: [Brief description]

**Key Insights**:
- ...

## Architecture Notes

### Data Flow
[How data moves through the system]

### Key Abstractions
[Important classes/functions and their roles]

### Design Decisions
[Notable implementation choices and their implications]

## Relevance to Lab Research

### Potential Reuse
- [ ] Can reuse [component] for [purpose]
- [ ] Can adapt [approach] for [paper]

### Learnings
- Implementation insight 1
- Implementation insight 2

### Gaps or Limitations
- ...

## Cross-References
- **Paper Summary**: reading_stack/summaries/PAPER-XXX.md
- **Related Repos**: REPO-YYY
- **Lab Papers Using**: papers/<paper-name>/
```

---

## Review Process

### Phase 1: Initial Scan (15-30 min)
1. Read README and documentation
2. Understand directory structure
3. Identify entry points
4. Note tech stack and dependencies

### Phase 2: Core Understanding (1-2 hours)
1. Trace main execution path
2. Identify key abstractions and data structures
3. Understand the core algorithm implementations
4. Note any clever or unusual patterns

### Phase 3: Method Deep-Dive (per method)
1. Locate implementation of specific paper methods
2. Document the implementation approach
3. Compare to paper description
4. Note any deviations or extensions

### Phase 4: Quality Assessment
1. Check for tests and their coverage
2. Assess documentation quality
3. Evaluate reproducibility (configs, seeds, data)
4. Consider reusability and adaptability

---

## Quality Criteria

### Code Quality Ratings

**Readability**
- Excellent: Clean, well-named, follows conventions
- Good: Mostly clear, occasional confusion
- Fair: Readable with effort, inconsistent style
- Poor: Difficult to follow, unclear intent

**Documentation**
- Thorough: Docstrings, comments, architecture docs
- Adequate: Key functions documented, some gaps
- Minimal: Sparse comments, relies on code clarity
- None: No documentation

**Tests**
- Comprehensive: High coverage, edge cases, integration
- Adequate: Core functionality tested
- Minimal: Few tests, basic coverage
- None: No tests

**Reproducibility**
- Easy: Clear instructions, configs, seeds, data access
- Possible: Can reproduce with some effort
- Difficult: Missing configs, unclear setup, data issues

---

## ID Conventions

- Repos: `REPO-001`, `REPO-002`, etc.
- Cross-reference papers using their `PAPER-XXX` IDs

IDs are assigned sequentially. When a repo corresponds to a paper, note the `PAPER-XXX` reference in the entry.

---

## Integration with Reading Stack

When reviewing a repo that corresponds to a paper:

1. **If paper not yet read**: Add to `reading_stack/inbox/` first
2. **Cross-reference**: Link REPO-XXX ↔ PAPER-XXX in both summaries
3. **Implementation notes**: Reference repo summary in paper's "Potential Connections" section

This creates a bidirectional link between theoretical understanding (paper) and practical implementation (code).
