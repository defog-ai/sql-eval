# Contributing Guidelines

Thank you for your interest in contributing to our project! We value your contributions and want to ensure a smooth and collaborative experience for everyone. Please take a moment to review the following guidelines.

## Table of Contents
- [Installation](#installation)
- [Linting](#linting)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Installation

Firstly, clone the repository where we store our database data and schema. Install all Python libraries listed in the `requirements.txt` file. Download the spacy model used in the NER heuristic for our [metadata-pruning method](https://github.com/defog-ai/sql-eval/blob/main/utils/pruning.py). Finally, install the library:
```bash
git clone https://github.com/defog-ai/defog-data.git
cd defog-data
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

## Linting

We use [black](https://black.readthedocs.io/en/stable/) for code formatting and linting. After installing it via pip, you can automatically lint your code with black by adding it as a pre-commit git hook:
```bash
pip install black
echo -e '#!/bin/sh\n#\n# Run linter before commit\nblack $(git rev-parse --show-toplevel)' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

## Testing

[_Quis probabit ipsa probationem?_](https://en.wikipedia.org/wiki/Quis_custodiet_ipsos_custodes%3F)

We have a comprehensive test suite that ensures the quality and reliability of our codebase. To run the python tests, you can use the following command:

```bash
pytest tests
```

Our CI excludes [tests/verify_questions.py](tests/verify_questions.py) as it depends on having a local postgres environment with the data loaded.

Please make sure that all tests pass before submitting your changes.

We also understand that some changes might not be easily testable with unit tests. In such cases, please provide a detailed description of your changes and how you tested them. We will review your changes and work with you to ensure that they are tested and verified.

## Submitting Changes

When submitting changes to this repository, please follow these steps:

- Fork the repository and create a new branch for your changes.
- Make your changes, following the coding style and best practices outlined here.
- Run the tests to ensure your changes don't introduce any regressions.
- Lint your code and [squash your commits](https://www.git-tower.com/learn/git/faq/git-squash) down to 1 single commit.
- Commit your changes and push them to your forked repository.
- Open a pull request to the main repository and provide a detailed description of your changes:
  - What problem you are trying to solve.
  - Alternatives considered and how you decided on which one to take
  - How you solved it.
  - How you tested your changes.
- Your pull request will be reviewed by our team, and we may ask for further improvements or clarifications before merging. Thank you for your contribution!