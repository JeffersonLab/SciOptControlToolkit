#!/bin/bash

REPO_PATH=$(git rev-parse --show-toplevel)/utests
echo 'Repo directory '$REPO_PATH

#for filename in $REPO_PATH/$PATTERN; do
for filename in ${REPO_PATH}/*.py; do
  echo 'Running utest: '$filename
  python $filename
done
