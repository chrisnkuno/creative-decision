#!/usr/bin/env bash
# The full gate: formatting, lint, strict types, invariant/behaviour/functional tests + coverage.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=cdm --cov-branch --cov-report=term-missing --cov-fail-under=90 "$@"
