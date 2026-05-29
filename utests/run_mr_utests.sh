#!/bin/bash
set -e

REPO_PATH=$(git rev-parse --show-toplevel)/utests
REPORT_DIR=$(git rev-parse --show-toplevel)/test-reports
echo 'Repo directory '$REPO_PATH
echo 'Report directory '$REPORT_DIR

mkdir -p "$REPORT_DIR"

# Discovers and runs all test_*.py files under utests/.
# --junitxml emits a report consumed by GitLab's MR Tests tab.
pytest "$REPO_PATH" \
  -v \
  --tb=short \
  --junitxml="$REPORT_DIR/junit.xml"