#!/bin/bash
set -e

python utests/utest_agents.py
python utests/utest_buffers.py
python utests/utest_envs_and_utils.py
python utests/utest_models.py
python utests/utest_registry.py