"""Minimal example: train TorchDEPO against the FEL digital twin for one
already-adapted continual-learning cycle.

This is a STARTING POINT, not a production orchestration service. It does
NOT run fel_dt's ContinualLearningDriver itself (that already has its own
full CLI -- `python -m fel_dt.driver.continual_learning --config ...` --
see the fel_dt repo) or loop the scan->adapt->train cycle indefinitely.
Point it at one cycle's already-written model_weights.pth + train_data.npz
(exactly what adapt_to_task's output_dir already contains right after a
cycle finishes -- e.g.
artifacts_naive_10task_regime_aware_v2/cycle_N/adaptation/), and it will:

  1. Build fel_dt's FELDigitalTwinBatchEnv (via Gymnasium's registry) around
     that checkpoint.
  2. Construct a TorchDEPO agent on it.
  3. Run train() a configured number of times, logging to --logdir.
  4. Save the trained actor.

Repeating this per cycle -- reconstructing the env (or calling its
set_model()) with each new cycle's outputs as the continual-learning driver
produces them -- is the natural next step toward the full synchronous
scan/adapt/train loop, once this base is proven out; not built here.

Usage:
    python -m jlab_opt_control.drivers.run_fel_digital_twin \\
        --model-path .../cycle_1/adaptation/model_weights.pth \\
        --recent-data-path .../cycle_1/adaptation/train_data.npz \\
        --base-minimums-path .../good_artifacts/task_0/minimums.pkl \\
        --base-maximums-path .../good_artifacts/task_0/maximums.pkl \\
        --vars-path .../configs/vars.yaml \\
        --n-train-calls 200
"""

import argparse
import logging

import gymnasium

import fel_dt.envs  # noqa: F401 -- import triggers Gymnasium registration
import jlab_opt_control.agents as agents

run_log = logging.getLogger("RunFELDigitalTwin")
run_log.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="A cycle's adaptation/model_weights.pth")
    parser.add_argument("--recent-data-path", required=True, help="That cycle's adaptation/train_data.npz")
    parser.add_argument("--base-minimums-path", required=True, help="e.g. good_artifacts/task_0/minimums.pkl")
    parser.add_argument("--base-maximums-path", required=True, help="e.g. good_artifacts/task_0/maximums.pkl")
    parser.add_argument("--vars-path", required=True, help="fel_dt's configs/vars.yaml")
    parser.add_argument("--delta-fraction", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--unroll-steps", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--n-train-calls", type=int, default=200)
    parser.add_argument("--logdir", default="/tmp/fel_digital_twin_depo")
    args = parser.parse_args(args)

    env = gymnasium.make(
        "FELDigitalTwin-Torch-Batch-v0",
        model_path=args.model_path, base_minimums_path=args.base_minimums_path,
        base_maximums_path=args.base_maximums_path, vars_path=args.vars_path,
        recent_data_path=args.recent_data_path, delta_fraction=args.delta_fraction,
        episode_length=args.episode_length,
    )

    agent = agents.make(
        "TorchDEPO-v0", env=env, logdir=args.logdir,
        unroll_steps=args.unroll_steps, discount=args.discount,
    )
    agent.rollout_batch_size = args.rollout_batch_size

    for i in range(args.n_train_calls):
        discounted_return = agent.train()
        if i % 10 == 0 or i == args.n_train_calls - 1:
            run_log.info(f"train() call {i}: discounted_return={discounted_return.item():.4f}")

    agent.save(post_fix="final")
    run_log.info(f"Done. Actor + logs written under {args.logdir}")


if __name__ == "__main__":
    main()
