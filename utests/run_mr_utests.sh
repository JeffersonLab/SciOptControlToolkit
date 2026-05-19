#!/bin/bash
set -e

REPO_PATH=$(git rev-parse --show-toplevel)/utests
REPORT_DIR=$(git rev-parse --show-toplevel)/test-reports
echo 'Repo directory '$REPO_PATH
echo 'Report directory '$REPORT_DIR

mkdir -p "$REPORT_DIR"

# Run all unit tests under utests/ via pytest.
# --junitxml emits a report consumed by GitLab's MR Tests tab.
# --tb=short keeps tracebacks readable in CI logs.
# -v prints per-test pass/fail lines.
pytest "$REPO_PATH" \
  -v \
  --tb=short \
  --junitxml="$REPORT_DIR/junit.xml"