#!/usr/bin/env python3
"""
Construct a registered agent/model/buffer through the real jlab_opt_control
registry and loudly surface any ERROR-level log line raised during
construction.

Why this exists: this codebase's cfg-shape validation (e.g. "does
activation_functions have the right length for this many hidden layers?")
only ever logs an error on mismatch — it never raises. A component registered
against the wrong packaged cfg file constructs "successfully" and runs with
a silently-wrong architecture. This shipped twice (see KNOWN_ISSUES.md, B2
and B3) before being caught by manual review. Run this immediately after
registering any agent/model/buffer.

Requires the project's runtime deps (TensorFlow, Gymnasium, ...) — activate
the project's conda env first: `conda activate jlab_opt_control_env`.

Usage:
    python check_cfg_shape.py agent  KerasX-v0
    python check_cfg_shape.py model  new_model-v0 [--state-dim 4] [--action-dim 2]
    python check_cfg_shape.py buffer NewBuffer-v0 [--state-dim 4] [--action-dim 2]
"""

import argparse
import logging
import sys
import tempfile


class CollectingHandler(logging.Handler):
    """Captures every ERROR+ log record emitted anywhere during construction."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


def check_agent(id, env_id):
    import gymnasium as gym
    import jlab_opt_control.agents as agents

    env = gym.make(env_id)
    logdir = tempfile.mkdtemp()
    obj = agents.make(id, env=env, logdir=logdir)
    return obj


def check_model(id, state_dim, action_dim):
    import numpy as np
    import jlab_opt_control.models as models

    logdir = tempfile.mkdtemp()
    min_action = -np.ones(action_dim, dtype=np.float32)
    max_action = np.ones(action_dim, dtype=np.float32)

    # Model constructors aren't uniform (actor-shaped vs. critic-shaped vs.
    # SINDy-shaped) — try the shapes that exist in this codebase today, in
    # order, and report which one actually worked.
    attempts = [
        ("actor-shaped (state_dim, action_dim, min_action, max_action, logdir)",
         dict(state_dim=state_dim, action_dim=action_dim,
              min_action=min_action, max_action=max_action, logdir=logdir)),
        ("critic-shaped (state_dim, action_dim, logdir)",
         dict(state_dim=state_dim, action_dim=action_dim, logdir=logdir)),
        ("SINDy-shaped (num_features_in, num_features_out, logdir)",
         dict(num_features_in=state_dim, num_features_out=action_dim, logdir=logdir)),
    ]
    errors = []
    for desc, kwargs in attempts:
        try:
            obj = models.make(id, **kwargs)
            print(f"  (constructed using the {desc} signature)")
            return obj
        except TypeError as e:
            errors.append(f"  tried {desc}: {e}")
            continue
    raise TypeError(
        "Could not construct with any known model constructor shape:\n" +
        "\n".join(errors) +
        "\nIf your model has a genuinely different signature, adapt this "
        "script's `attempts` list rather than skipping the check."
    )


def check_buffer(id, state_dim, action_dim):
    import jlab_opt_control.buffers as buffers

    logdir = tempfile.mkdtemp()
    obj = buffers.make(id, state_dim=state_dim, action_dim=action_dim,
                        logdir=logdir, buffer_size=1000)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["agent", "model", "buffer"])
    parser.add_argument("id", help="registry id, e.g. KerasTD3-v0 or actor_fcnn-v0")
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--env", default="Pendulum-v1",
                         help="Gym env id to use when checking an agent (default: Pendulum-v1)")
    args = parser.parse_args()

    handler = CollectingHandler()
    logging.getLogger().addHandler(handler)

    print(f"Constructing {args.kind} '{args.id}' through the real registry...")
    try:
        if args.kind == "agent":
            check_agent(args.id, args.env)
        elif args.kind == "model":
            check_model(args.id, args.state_dim, args.action_dim)
        else:
            check_buffer(args.id, args.state_dim, args.action_dim)
    except Exception as e:
        print(f"\nFAIL — construction raised {type(e).__name__}: {e}")
        sys.exit(1)

    if handler.records:
        print(f"\nFAIL — construction 'succeeded' but logged "
              f"{len(handler.records)} ERROR-level message(s) that this "
              f"codebase's validation would otherwise swallow silently:")
        for r in handler.records:
            print(f"  {r}")
        print("\nThis almost always means the registered cfg's shape doesn't "
              "match what this class's constructor validates — see "
              "agent-conventions/model.md's activation-count table, or the "
              "equivalent check in your class's __init__.")
        sys.exit(1)

    print(f"\nPASS — '{args.id}' constructed cleanly with no ERROR-level log output.")


if __name__ == "__main__":
    main()
