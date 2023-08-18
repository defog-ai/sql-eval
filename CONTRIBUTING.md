# Contributing Guidelines

Thank you for your interest in contributing to our project! We value your contributions and want to ensure a smooth and collaborative experience for everyone. Please take a moment to review the following guidelines.

## Table of Contents
- [Linting](#linting)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Linting

We use [black](https://black.readthedocs.io/en/stable/) for code formatting and linting. After installing it via pip, you can automatically lint your code with black by adding it as a pre-commit git hook:
```bash
pip install black
echo -e '#!/bin/sh\n#\n# Run linter before commit\nblack $(git rev-parse --show-toplevel)' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

## Testing

[_Quis probabit ipsa probationem?_](https://en.wikipedia.org/wiki/Quis_custodiet_ipsos_custodes%3F)

We have a comprehensive test suite that ensures the quality and reliability of our codebase. To run the tests, you can use the following command:

```bash
pytest tests
```

Please make sure that all tests pass before submitting your changes.

## Submitting Changes

When submitting changes to this repository, please follow these steps:

- Fork the repository and create a new branch for your changes.
- Make your changes, following the coding style and best practices outlined here.
- Run the tests to ensure your changes don't introduce any regressions.
- Lint your code and [squash your commits](https://www.git-tower.com/learn/git/faq/git-squash) down to 1 single commit.
- Commit your changes and push them to your forked repository.
- Open a pull request to the main repository and provide a detailed description of your changes.
- Your pull request will be reviewed by our team, and we may ask for further improvements or clarifications before merging. Thank you for your contribution!