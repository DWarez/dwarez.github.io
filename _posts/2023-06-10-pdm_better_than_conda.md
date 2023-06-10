---
layout: post
title: Use PDM instead of Conda
categories: [DevOps]
tags: [devops, pdm, conda, pytest]
date: 2023-06-10 +0200
pin: true
---
I gotta be honest. I don't like [Conda](https://docs.conda.io/en/latest/) that much. It's very slow and I don't like the fact that, even if I know the specific name of the package I want to install, I first must search in which repository it's contained.
I gave [Mamba](https://mamba.readthedocs.io/en/latest/) a try. Even if it seems a bit faster, it breaks everytime I try to uninstall a package or remove an environment.

In the last weeks, I've been using [PDM](https://pdm.fming.dev/latest/) and I must say that I was pleasantly surprised by its speed, configurability, and ease of use.

## A simple init
Let's say you want to create a fresh environment for your project.

First, create a new virtual env[^1]:

```bash
python3 -m venv .venv
```

Now, simply run the PDM project init function:
```bash
pdm init
```

Select the virtual environment we just created as the Python interpreter to use, and answer to the following prompts.

If you want to install a package, run:
```bash
pdm add [package_name]
```

PDM will keep track of all the packages you install while checking for conflicts and incompatibility issues, using the `pdm.lock` file. Furthermore, it will update the `pyproject.toml` file, with additional, human readable information.



## Integrate CI/CD tools
I highly suggest using DevOps tools, which can be easily integrated with PDM. I personally use [pre-commit](https://pre-commit.com/) to set up the pre-commit and pre-push hooks, which will trigger a set of heterogeneous checks before committing to the branch.

Here's an example configuration file for pre-commit with PDM:

```yaml
default_language_version:
  python: python3.10
fail_fast: true
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.5.0  # Use the ref you want to point at
    hooks:
      - id: check-added-large-files
        exclude: pdm.lock
      - id: check-yaml
      - id: check-ast
      - id: check-case-conflict
      - id: no-commit-to-branch
        args: ['--branch', 'main', '--branch' , 'develop', '--pattern', '(?!^(feature|refactor|bugfix|hotfix|release|document|ignored)/.*)']
        stages: [push]
  - repo: local
    hooks:
      - id: isort
        name: isort
        stages: [commit]
        language: system
        entry: pdm run isort
        types: [python]

      - id: black
        name: black
        stages: [commit]
        language: system
        entry: pdm run black
        types: [python]

      - id: flake8
        name: flake8
        stages: [commit]
        language: system
        entry: pdm run flake8
        types: [python]
        exclude: setup.py

      - id: mypy
        name: mypy
        description: ''
        stages: [commit]
        entry: pdm run mypy
        language: python
        'types_or': [python, pyi]
        args: ["--ignore-missing-imports", "--scripts-are-modules", "--explicit-package-bases"]
        require_serial: true
        additional_dependencies: []
        minimum_pre_commit_version: '2.9.2'

      - id: pytest
        name: pytest
        stages: [commit]
        language: system
        entry: pdm run pytest
        types: [python]
        args: [-x]
        pass_filenames: false

      - id: pytest-cov
        name: pytest-cov
        stages: [push]
        language: system
        entry: pdm run pytest --cov --cov-fail-under=100
        types: [python]
        pass_filenames: false
```

While you can configure the tools at the end of the project's  `pyproject.toml`, like so:

```yaml
[tool.black]
line-length = 88
target-version = ['py310']
preview = true

[tool.isort]
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
line_length = 88
profile = "black"

[tool.mypy]
# files = [""]
packages = ["src", "tests"]
plugins = ["pydantic.mypy", "sqlmypy"]
disallow_untyped_defs = "True"
disallow_any_unimported = "True"
ignore_missing_imports = "True"
no_implicit_optional = "True"
check_untyped_defs = "True"
warn_return_any = "True"
warn_unused_ignores = "True"
show_error_codes = "True"
exclude = ['volumes/', "alembic/", "scripts/", "docs/", "settings/", ".vscode/", ".venv/", ".pytest_cache/", ".mypy_cache/", ".gitlab/", ".github/", ".devcontainer/", "Docker/", "dashboards/"]
disable_error_code = ["str-bytes-safe", "no-any-unimported"]
```

Additional configuration files my be used, like a `.coveragerc` file for your pytests:

```yaml
[run]
source =
       src
       tests

[report]
exclude_lines =
    # Have to re-enable the standard pragma
    pragma: no cover

    # Don't complain about missing debug-only code:
    def __repr__
    if self\.debug

    # Don't complain if tests don't hit defensive assertion code:
    raise AssertionError
    raise NotImplementedError

    # Don't complain if non-runnable code isn't run:
    if 0:
    if __name__ == .__main__.:

```

## Copy the setup to another location
If you want to copy this setup to another location, simply copy the configuration files (like `pyproject.toml`) and run

```bash
pdm use         # for selecting the Python interpreter to use
pdm install     # for installing the dependencies described by pyproject.toml
```

That's all folks. Bye 🤖

[^1]: I know that some people dislike using virtual envs like this. I personally find them very helpful.