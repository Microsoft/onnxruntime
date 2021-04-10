# Python Code Checks

Python code checks are run by this [CI build](../azure-pipelines/python-checks-ci-pipeline.yml).
Here are instructions on how to run them manually.

## Prerequisites

Install requirements.

From the repo root, run:

`$ python -m pip install -r tools/ci_build/github/python_checks/requirements.txt`

## Flake8

From the repo root, run:

`$ python -m flake8 --config .flake8`
