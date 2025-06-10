#!/usr/bin/env bash

# scripts/setup_local_env.sh
# ------------------------------------------------------------------------------
# Utility script to create a local Python virtual environment (".venv") and
# install the project dependencies using Poetry (preferred) or pip fall-back.
# This is meant for contributors who want to run GoldenSignalsAI outside of the
# full Docker stack. It ensures a reproducible local dev environment.
# ------------------------------------------------------------------------------

set -euo pipefail

# Colours for terminal output
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

info()   { echo -e "${GREEN}[INFO]${NC} $1"; }
fail()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect Python executable (prefer 3.11, fall back to system python3)
PYTHON_BIN="$(command -v python3.11 || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  fail "Python 3.11+ is required but not found in PATH."
fi

info "Using Python interpreter: ${PYTHON_BIN}"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
  info "Creating virtual environment in .venv ..."
  "${PYTHON_BIN}" -m venv .venv
fi

# Activate venv
source .venv/bin/activate
info "Activated virtual environment (.venv)"

# Upgrade pip and install Poetry if not present
pip install --quiet --upgrade pip
if ! command -v poetry >/dev/null 2>&1; then
  info "Installing Poetry ..."
  pip install --quiet poetry
fi

# Ensure Poetry uses in-project venv
poetry config virtualenvs.in-project true --local

# Install dependencies
if [[ -f "pyproject_v3.toml" ]]; then
  info "Exporting dependencies from pyproject_v3.toml and installing via pip ..."
  tmp_req="$(mktemp)"
  # Use Poetry to export requirements without hashes
  poetry export --without-hashes --format=requirements.txt -o "$tmp_req" --directory . --file pyproject_v3.toml || {
    fail "Failed to export requirements from pyproject_v3.toml";
  }
  pip install --quiet -r "$tmp_req"
  rm "$tmp_req"
else
  if [[ -f "pyproject.toml" ]]; then
    info "Installing dependencies from pyproject.toml via Poetry ..."
    poetry install --no-root --no-interaction --sync --with=dev
  elif [[ -f "requirements.txt" ]]; then
    info "Installing dependencies from requirements.txt via pip ..."
    pip install --quiet -r requirements.txt
  else
    fail "No dependency manifest (pyproject.toml or requirements.txt) found."
  fi
fi

info "Local environment setup complete. To activate it later run:"
info "  source .venv/bin/activate"
