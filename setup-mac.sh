#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${LAB_PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        LAB_PYTHON="$PWD/.venv/bin/python"
    else
        LAB_PYTHON=python3.12
    fi
fi

if ! command -v "$LAB_PYTHON" >/dev/null 2>&1; then
    echo "Python 3.12 was not found. Install it, reopen Terminal, and try again." >&2
    echo "See README.md, checkpoint 2. The macOS system Python may be older." >&2
    exit 2
fi

exec "$LAB_PYTHON" scripts/setup_environment.py "$@"
