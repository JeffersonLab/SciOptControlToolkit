#!/bin/bash
set -e

REPO_PATH=$(git rev-parse --show-toplevel)/utests
echo 'Repo directory '$REPO_PATH

pytest \
  "$REPO_PATH/test_agents.py" \
  "$REPO_PATH/test_buffers.py" \
  "$REPO_PATH/test_envs_and_utils.py" \
  "$REPO_PATH/test_models.py" \
  "$REPO_PATH/test_registry.py" \
  -v \
  --tb=short