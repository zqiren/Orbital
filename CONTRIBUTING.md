# Contributing to Orbital

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/` and `cd web && npm test`)
5. Commit with a clear message
6. Push and open a Pull Request

## Contributor License Agreement

By submitting a pull request, you agree to license your contribution under the project's GPL-3.0 license and grant the project maintainers the right to use, modify, and relicense your contribution under any license. This is necessary to preserve the maintainers' ability to offer the software under alternative licenses for commercial use.

If you do not agree with these terms, please do not submit a pull request.

## Code Style

- Python: Follow existing patterns. Type hints preferred.
- TypeScript/React: Follow existing patterns. Functional components with hooks.

## Testing

- Every bug fix must include a regression test in `tests/regression/`
- Run `pytest tests/` for Python tests
- Run `cd web && npm test` for frontend tests

## Reporting Issues

Please include:
- OS and version
- Python and Node.js versions
- Steps to reproduce
- Expected vs actual behavior
