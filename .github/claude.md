# Claude for GitHub Integration Guide

## Overview
Claude for GitHub is an AI-powered assistant that helps with code reviews, issue management, and development workflows directly within GitHub.

## Features & Usage

### 1. **Automated Code Reviews**
Claude can automatically review pull requests and provide feedback.

**How to use:**
- Create a pull request
- Claude will automatically analyze the changes
- Look for Claude's comments with suggestions for:
  - Code quality improvements
  - Security vulnerabilities
  - Performance optimizations
  - Best practices

### 2. **Issue Analysis & Solutions**
Claude can help solve issues by analyzing the problem and suggesting fixes.

**Commands in issues:**
- `@claude help` - Get help with the current issue
- `@claude analyze` - Deep analysis of the problem
- `@claude suggest fix` - Get code suggestions
- `@claude explain` - Explain complex code or errors

### 3. **Pull Request Assistance**
Claude can help improve your PRs.

**Commands in PRs:**
- `@claude review` - Request a detailed code review
- `@claude security` - Security-focused review
- `@claude performance` - Performance analysis
- `@claude tests` - Suggest test cases
- `@claude docs` - Help with documentation

### 4. **Automated Workflows**
You can trigger Claude in GitHub Actions.

**Example workflow:**
```yaml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  claude-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Claude AI Review
        uses: anthropic/claude-github-action@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          claude-api-key: ${{ secrets.CLAUDE_API_KEY }}
          review-type: 'comprehensive'
          include-security: true
          include-performance: true
          include-tests: true
```

## Configuration for Your Project

### 1. **Create Claude Configuration**
Create `.github/claude.yml`:

```yaml
# Claude for GitHub Configuration
version: 1

# Review settings
reviews:
  auto_review: true
  review_drafts: false

  # What to check
  checks:
    - security
    - performance
    - best_practices
    - documentation
    - tests

  # Language-specific settings
  languages:
    python:
      style_guide: "black"
      max_line_length: 100
      type_checking: true

    typescript:
      style_guide: "eslint"
      strict_mode: true

# Issue assistance
issues:
  auto_label: true
  suggest_solutions: true

  # Labels Claude can add
  labels:
    - "needs-review"
    - "security"
    - "performance"
    - "bug"
    - "enhancement"

# Custom prompts
prompts:
  review_focus: |
    Focus on:
    1. Security vulnerabilities
    2. Performance optimizations
    3. React best practices
    4. TypeScript type safety
    5. WebSocket connection stability

  project_context: |
    This is a FinTech trading signals application with:
    - React/TypeScript frontend
    - FastAPI Python backend
    - Real-time WebSocket connections
    - AI-powered analysis with multiple LLMs
```

### 2. **Add to Existing Workflows**
Update your `.github/workflows/pr-checks.yml`:

```yaml
- name: Claude PR Analysis
  if: github.event_name == 'pull_request'
  run: |
    echo "@claude review" >> $GITHUB_STEP_SUMMARY
```

### 3. **Create Issue Templates with Claude**
Create `.github/ISSUE_TEMPLATE/bug_report_claude.md`:

```markdown
---
name: Bug Report (with Claude)
about: Create a bug report with AI assistance
title: '[BUG] '
labels: 'bug, needs-claude-review'
assignees: ''
---

## Description
<!-- Describe the bug -->

## To Reproduce
<!-- Steps to reproduce -->

## Expected behavior
<!-- What should happen -->

## Actual behavior
<!-- What actually happens -->

---
@claude analyze this bug and suggest potential fixes
```

## Best Practices

### 1. **Effective Commands**
- Be specific: `@claude review the WebSocket connection logic`
- Ask for examples: `@claude show me how to fix this ESLint error`
- Request explanations: `@claude explain why this might cause a memory leak`

### 2. **Code Review Workflow**
1. Create PR with clear description
2. Let Claude do initial review
3. Address Claude's feedback
4. Request human review

### 3. **Issue Resolution**
1. Create detailed issue
2. Tag `@claude analyze`
3. Review Claude's suggestions
4. Implement and reference Claude's advice in PR

## Advanced Usage

### 1. **Custom Commands**
Create `.github/claude-commands.yml`:

```yaml
commands:
  fix-lint:
    description: "Fix linting issues"
    action: |
      Analyze the linting errors and provide fixes for:
      - ESLint violations
      - TypeScript errors
      - Import organization

  optimize-chart:
    description: "Optimize chart performance"
    action: |
      Review the AITradingChart component and suggest:
      - Rendering optimizations
      - Memory leak fixes
      - Performance improvements
```

### 2. **Automated Fix PRs**
Claude can create PRs with fixes:

```yaml
# In issue comment
@claude create-pr-with-fix
```

### 3. **Learning from Your Codebase**
Claude learns from your:
- Coding patterns
- PR reviews
- Issue resolutions
- Documentation

## Security Considerations

1. **Never expose sensitive data** in issues/PRs
2. **Use environment variables** for API keys
3. **Review Claude's suggestions** before implementing
4. **Don't auto-merge** Claude's PRs without review

## Getting Started

1. **Test Claude** on a simple issue:
   ```
   @claude help me understand the WebSocket reconnection logic
   ```

2. **Try on a PR**:
   ```
   @claude review this PR focusing on security
   ```

3. **Create an issue** for your ESLint errors:
   ```
   Title: Fix ESLint errors in frontend
   Body: @claude analyze these ESLint errors and provide fixes
   ```

## Useful for Your Current Tasks

Given your todo list, Claude can help with:

1. **ESLint Fixes**: Create an issue and ask Claude for fixes
2. **WebSocket Review**: Ask Claude to review the WebSocket implementation
3. **Chart Optimization**: Get suggestions for AITradingChart improvements
4. **Test Writing**: Ask Claude to suggest test cases

## Tips

- Use Claude for code reviews before human reviews
- Ask Claude to explain complex parts of the codebase
- Let Claude suggest refactoring opportunities
- Use Claude to generate documentation
- Ask Claude for best practices specific to your tech stack
