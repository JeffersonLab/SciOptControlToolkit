#!/bin/bash
set -e

python utests/utest_agents.py
python utests/utest_buffers.py
pyhton utests/utest_envs_and_utils.py
pyhton utests/utest_models.py
python utests/utest_registry.py