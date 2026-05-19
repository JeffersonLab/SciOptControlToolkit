#!/bin/bash
set -e

REPO_PATH=$(git rev-parse --show-toplevel)/utests
echo 'Repo directory '$REPO_PATH

pytest \
  "$REPO_PATH/utest_agents.py" \
  "$REPO_PATH/utest_buffers.py" \
  "$REPO_PATH/utest_envs_and_utils.py" \
  "$REPO_PATH/utest_models.py" \
  "$REPO_PATH/utest_registry.py" \
  -v \
  --tb=short