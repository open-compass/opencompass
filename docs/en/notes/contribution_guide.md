# Contributing to OpenCompass

- [Contributing to OpenCompass](#contributing-to-opencompass)
  - [Workflow](#workflow)
  - [Code style](#code-style)
    - [Python](#python)
  - [Pre-commit Hook](#pre-commit-hook)

Thanks for your interest in contributing to OpenCompass! All kinds of contributions are welcome, including but not limited to the following.

- Fix typo or bugs
- Add documentation or translate the documentation into other languages
- Add new features and components

## Workflow

We recommend the potential contributors follow this workflow for contribution.

1. Fork and pull the latest OpenCompass repository, follow [get started](https://OpenCompass.readthedocs.io/en/latest/get_started.html) to setup the environment.
2. Checkout a new branch (**do not use the master or dev branch** for PRs)

```bash
git checkout -b xxxx # xxxx is the name of new branch
```

3. Edit the related files follow the code style mentioned below
4. Use [pre-commit hook](https://pre-commit.com/) to check and format your changes.
5. Commit your changes
6. Create a PR with related information

## Code style

### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](https://github.com/PyCQA/flake8): A wrapper around some linter tools.
- [isort](https://github.com/timothycrosley/isort): A Python utility to sort imports.
- [yapf](https://github.com/google/yapf): A formatter for Python files.
- [codespell](https://github.com/codespell-project/codespell): A Python utility to fix common misspellings in text files.
- [mdformat](https://github.com/executablebooks/mdformat): Mdformat is an opinionated Markdown formatter that can be used to enforce a consistent style in Markdown files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of yapf and isort can be found in [setup.cfg](https://github.com/open-mmlab/OpenCompass/blob/main/setup.cfg).

## Pre-commit Hook

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`,
fixes `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, sorts `requirements.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](xxxxxxx).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.

> Before you create a PR, make sure that your code lints and is formatted by yapf.
