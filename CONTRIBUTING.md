# Contributing to GoldenSignalsAI

## Welcome!
We appreciate your interest in contributing to GoldenSignalsAI. This document provides guidelines for contributing to the project.

## Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Collaborate openly and transparently

## Development Setup
1. Fork the repository
2. Clone your fork
3. Create a virtual environment
```bash
conda create -n goldensignalsai-dev python=3.10
conda activate goldensignalsai-dev
```
4. Install dependencies
```bash
poetry install --no-root
```

## Contribution Workflow
1. Create a new branch
```bash
git checkout -b feature/your-feature-name
```
2. Make your changes
3. Run tests
```bash
poetry run pytest --cov=./
```
4. Run linters
```bash
black .
flake8 .
```
5. Commit your changes
```bash
git add .
git commit -m "Description of changes"
```
6. Push to your fork
```bash
git push origin feature/your-feature-name
```
7. Open a Pull Request

## Contribution Areas
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations
- Additional agent implementations

## Code Guidelines
- Follow PEP 8 style guide
- Write comprehensive unit tests
- Document new functions and classes
- Maintain modular architecture
- Use type hints

## Reporting Issues
- Use GitHub Issues
- Provide detailed description
- Include reproduction steps
- Specify Python and dependency versions

## Pull Request Process
1. Ensure all tests pass
2. Update documentation
3. Add comments explaining complex logic
4. Get approval from maintainers

## Versioning & Releases
- Follow [Semantic Versioning](https://semver.org/)
- Tag releases (e.g., `v1.1.0`) for changelog and deployment
- GitHub Actions automates tests and deployment (see `.github/workflows/ci.yml`)

## Contact
For questions, please open a GitHub Issue or contact the maintainer.

Thank you for contributing to GoldenSignalsAI!
