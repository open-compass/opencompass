# Contributing to OpenCompass

- [Contributing to OpenCompass](#contributing-to-opencompass)
  - [What is PR](#what-is-pr)
  - [Basic Workflow](#basic-workflow)
  - [Procedures in detail](#procedures-in-detail)
    - [1. Get the most recent codebase](#1-get-the-most-recent-codebase)
    - [2. Checkout a new branch from `main` branch](#2-checkout-a-new-branch-from-main-branch)
    - [3. Commit your changes](#3-commit-your-changes)
    - [4. Push your changes to the forked repository and create a PR](#4-push-your-changes-to-the-forked-repository-and-create-a-pr)
    - [5. Discuss and review your code](#5-discuss-and-review-your-code)
    - [6.  Merge your branch to `main` branch and delete the branch](#6--merge-your-branch-to-main-branch-and-delete-the-branch)
  - [Code style](#code-style)
    - [Python](#python)
  - [About Contributing Test Datasets](#about-contributing-test-datasets)

Thanks for your interest in contributing to OpenCompass! All kinds of contributions are welcome, including but not limited to the following.

- Fix typo or bugs
- Add documentation or translate the documentation into other languages
- Add new features and components

## What is PR

`PR` is the abbreviation of `Pull Request`. Here's the definition of `PR` in the [official document](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) of Github.

```
Pull requests let you tell others about changes you have pushed to a branch in a repository on GitHub. Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the base branch.
```

## Basic Workflow

1. Get the most recent codebase
2. Checkout a new branch from `main` branch.
3. Commit your changes ([Don't forget to use pre-commit hooks!](#3-commit-your-changes))
4. Push your changes and create a PR
5. Discuss and review your code
6. Merge your branch to `main` branch

## Procedures in detail

### 1. Get the most recent codebase

- When you work on your first PR

  Fork the OpenCompass repository: click the **fork** button at the top right corner of Github page
  ![avatar](https://github.com/open-compass/opencompass/assets/22607038/851ed33d-02db-49c9-bf94-7c62eee89eb2)

  Clone forked repository to local

  ```bash
  git clone git@github.com:XXX/opencompass.git
  ```

  Add source repository to upstream

  ```bash
  git remote add upstream git@github.com:InternLM/opencompass.git
  ```

- After your first PR

  Checkout the latest branch of the local repository and pull the latest branch of the source repository.

  ```bash
  git checkout main
  git pull upstream main
  ```

### 2. Checkout a new branch from `main` branch

```bash
git checkout main -b branchname
```

### 3. Commit your changes

- If you are a first-time contributor, please install and initialize pre-commit hooks from the repository root directory first.

  ```bash
  pip install -U pre-commit
  pre-commit install
  ```

- Commit your changes as usual. Pre-commit hooks will be triggered to stylize your code before each commit.

  ```bash
  # coding
  git add [files]
  git commit -m 'messages'
  ```

  ```{note}
  Sometimes your code may be changed by pre-commit hooks. In this case, please remember to re-stage the modified files and commit again.
  ```

### 4. Push your changes to the forked repository and create a PR

- Push the branch to your forked remote repository

  ```bash
  git push origin branchname
  ```

- Create a PR
  ![avatar](https://github.com/open-compass/opencompass/assets/22607038/08feb221-b145-4ea8-8e20-05f143081604)

- Revise PR message template to describe your motivation and modifications made in this PR. You can also link the related issue to the PR manually in the PR message (For more information, checkout the [official guidance](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)).

- You can also ask a specific person to review the changes you've proposed.

### 5. Discuss and review your code

- Modify your codes according to reviewers' suggestions and then push your changes.

### 6. Merge your branch to `main` branch and delete the branch

- After the PR is merged by the maintainer, you can delete the branch you created in your forked repository.

  ```bash
  git branch -d branchname # delete local branch
  git push origin --delete branchname # delete remote branch
  ```

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

## About Contributing Test Datasets

- Submitting Test Datasets
  - Please implement logic for automatic dataset downloading in the code; or provide a method for obtaining the dataset in the PR. The OpenCompass maintainers will follow up accordingly. If the dataset is not yet public, please indicate so.
- Submitting Data Configuration Files
- Provide a README in the same directory as the data configuration. The README should include, but is not limited to:
  - A brief description of the dataset
  - The official link to the dataset
  - Some test examples from the dataset
  - Evaluation results of the dataset on relevant models
  - Citation of the dataset
- (Optional) Summarizer of the dataset
- (Optional) If the testing process cannot be achieved simply by concatenating the dataset and model configuration files, a configuration file for conducting the test is also required.
- (Optional) If necessary, please add a description of the dataset in the relevant documentation sections. This is very necessary to help users understand the testing scheme. You can refer to the following types of documents in OpenCompass:
  - [Circular Evaluation](../advanced_guides/circular_eval.md)
  - [Code Evaluation](../advanced_guides/code_eval.md)
  - [Contamination Assessment](../advanced_guides/contamination_eval.md)
