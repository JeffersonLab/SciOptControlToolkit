# Recipe: new driver

There is exactly one existing driver, `drivers/run_continuous.py`. This
recipe generalizes its *shape*, not every specific flag — treat the overall
pattern as load-bearing, and treat any single flag/threshold as incidental
unless your new driver needs the same behavior for the same reason. If
something here doesn't fit what you're building, say so rather than forcing
the fit.

## The shape

1. **Parse CLI args** with `argparse` — one flag per overridable setting,
   `type=os.path.abspath` for any `--xxx_cfg` path override (so relative
   paths resolve correctly regardless of the caller's working directory).
2. **Build the environment and agent through their registries** — never by
   importing a class directly. `run_continuous.py`'s `create_and_configure_env()`
   dispatches across `gym.envs.registry`, `jlab_opt_control.envs`, and
   (optionally, if installed) an external namespace like `paces_envs`, in
   that order — first registry containing the id wins. If your driver only
   ever targets `jlab_opt_control` envs, you can skip the multi-registry
   dispatch and just call `jlab_opt_control.envs.make(...)` directly.
3. **Forward only the kwargs that were actually set.** `run_opt()` builds an
   `agent_kwargs` dict and only adds a key when the corresponding CLI value
   isn't `None` — this is what lets an agent's own cfg-driven defaults
   survive when the caller doesn't override them (see `conventions.md`'s cfg
   precedence). Don't unconditionally forward every parsed arg.
4. **Loop, log, checkpoint.** Each step: get an action, step the env, push
   `(state, action, reward, next_state, terminate)` into `agent.memory()`,
   call `agent.train()`. Log scalars to TensorBoard via
   `tf.summary.scalar(name, data=value, step=...)` — always pass an explicit
   `step`. Gate checkpointing on a meaningful improvement (in
   `run_continuous.py`: average *inference* reward improving by a
   configurable fraction), not on training reward alone, since training
   reward is noisy and includes exploration.
5. **Match the existing output-directory contract** so downstream tooling
   (TensorBoard, `utils/benchmark_analysis.py`, notebooks) keeps working:

   | Path | Contents |
   |---|---|
   | `<logdir>/cfgs/` | Copies of every component's resolved cfg (via each `save_cfg()`) |
   | `<logdir>/models/<postfix>/` | Saved weights, one subfolder per checkpoint |
   | `<logdir>/buffers/buffer.npy` | Replay buffer snapshot |
   | `<logdir>/metrics/` | TensorBoard event files |
   | `<logdir>/results.npy` | Raw per-episode reward array |

If your driver is a *batch/sweep* over existing single-run drivers rather
than a new training loop (e.g. running several agents across several envs
for comparison), don't reimplement the loop — call the existing driver's
entry point in a loop instead, the way `utils/benchmark_analysis.py` calls
`run_opt(...)` repeatedly across `agents` × `environments` from its own cfg
file. That file is the template for a sweep-style driver; `run_continuous.py`
is the template for a single-run training/inference driver.

## Test

There's no dedicated "driver" test file or mixin — `utests/test_baselines.py`
exercises `run_continuous.main()` end-to-end with real (if short) training
runs across several agent/env/buffer combinations; follow its pattern
(`args = [...]; main(args)`) to add coverage for a new driver, and expect it
to be exercised manually or via the full MR suite rather than the fast
branch script (see `conventions.md`'s testing table).
