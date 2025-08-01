# Claude Instructions for GoldenSignalsAIv4

## 🚨 IMPORTANT: Development Philosophy & SOP
1. **Start Simple**: Always fix the immediate issue before adding features
2. **Don't Over-Engineer**: Ask "What's the simplest solution?" before building
3. **Incremental Changes**: Make small, testable improvements
4. **Fix, Don't Replace**: Improve existing code rather than parallel implementations
5. **User First**: Focus on solving the user's actual problem, not what you think they need

## Standard Operating Procedure (SOP) for Coding

### Before Starting Any Task
1. **Read Existing Code First**: Always check what already exists before creating new files
2. **Use TodoWrite Tool**: Plan tasks for anything requiring 3+ steps
3. **Check for Patterns**: Follow existing code conventions and patterns
4. **Verify Dependencies**: Ensure required libraries are already in package.json/requirements.txt

### During Development
1. **No Mock Data**: Only use real data from APIs, never simulate with Math.random()
2. **Fix Root Causes**: Don't work around errors - fix them at the source
3. **Single Responsibility**: Each function/component should do one thing well
4. **Type Safety**: Always define TypeScript types, never use `any` without good reason
5. **Error Handling**: Show clear error messages to users, never fail silently

### Code Quality Standards
1. **Linting**: Run `npm run lint` and `npm run typecheck` before committing
2. **Python**: Run `black src/` and `isort src/` before committing
3. **No Console Logs**: Remove all console.log statements in production code
4. **Clean Imports**: Remove unused imports immediately
5. **Meaningful Names**: Functions and variables should be self-documenting

### Testing & Validation
1. **Test Locally**: Always test changes before marking tasks complete
2. **Check All Affected Files**: One change can break multiple components
3. **Verify WebSocket Connections**: Test real-time features thoroughly
4. **Run Quality Checks**: Use pre-commit hooks (`pre-commit run --all-files`)

### Git & GitHub Workflow
1. **Commit Messages**: Use conventional commits (feat:, fix:, docs:, etc.)
2. **Small Commits**: Each commit should be one logical change
3. **Branch Names**: Use descriptive names (fix/eslint-errors, feat/ai-predictions)
4. **Pull Requests**: Reference issue numbers, provide clear descriptions
5. **Never Commit Secrets**: Use environment variables and GitHub secrets

### Common Pitfalls to Avoid
1. **Creating Duplicate Files**: Always search for existing implementations first
2. **Using Node.js Libraries in Browser**: Winston, fs, path won't work in frontend
3. **Ignoring TypeScript Errors**: They often reveal runtime issues
4. **Assuming Context**: Components may be used outside their providers
5. **Over-Architecting**: Start with the simplest solution that works

### When Stuck or Blocked
1. **Check Error Messages**: Read the full error, not just the first line
2. **Verify File Paths**: Ensure imports and file references are correct
3. **Check Network Tab**: Verify API endpoints exist and respond
4. **Review Recent Changes**: Use `git diff` to see what changed
5. **Ask for Help**: Create detailed GitHub issues with full context

## Memorized By Claude

### Claude Session Insights
- As an AI system working on this project, I recognize the importance of consistent development practices
- Always prioritize code quality, user experience, and incremental improvements
- Maintain a balance between technical excellence and practical solutions
- Focus on building a reliable, professional trading platform

### Memorized Best Practices
- Start with the simplest solution that solves the user's problem
- Avoid over-engineering and unnecessary complexity
- Prioritize fixing root causes over creating workarounds
- Maintain clean, type-safe, and well-documented code
- Use real data and avoid mock implementations
- Implement comprehensive error handling
- Follow consistent coding standards across the project

### Claude Memory Snapshots
- Compress data and operations where possible to optimize performance and reduce resource usage
