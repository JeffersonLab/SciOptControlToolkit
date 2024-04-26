#!/bin/bash

REPO_PATH=$(git rev-parse --show-toplevel)/utests
echo 'Repo directory '$REPO_PATH
PATTERN="*.py"

for filename in "$REPO_PATH"/$PATTERN; do
  echo 'Running utest: '$filename
  python $filename
done
