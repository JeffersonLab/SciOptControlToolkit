# Recipe: new environment

Envs are the one kind that doesn't get a `jlab_opt_control.core` base class
— subclass Gymnasium's `gym.Env` directly. Assumes you've read
`conventions.md`.

## 1. Implement

`envs/new_env.py`, subclassing `gymnasium.Env`. Required:
`__init__`, `step(action) -> (obs, reward, terminated, truncated, info)`,
`reset() -> (obs, info)`, plus `self.action_space` / `self.observation_space`
set in `__init__` (Gymnasium `spaces.Box`, matching the rest of this
package's continuous-control focus — every agent here asserts
`"Box" in str(type(env.action_space))`).

Unlike agents/models/buffers, envs generally take plain constructor kwargs
rather than a packaged `.cfg` file — see `envs/circle_env.py`
(`rdm_reset_mode`, `statefull`, `max_episode_steps`). A `.cfg` file is
optional: `drivers/run_continuous.py`'s `create_and_configure_env()` only
forwards `cfg=env_cfg` when `--env_cfg` is actually passed, so add one only
if your env genuinely benefits from CLI-overridable JSON config; a plain
Python constructor kwarg is the simpler default for anything else.

## 2. Register

In `envs/__init__.py`:

```python
from jlab_opt_control.envs.new_env import NewEnv
...
register(
    id='<Prefix>-NewEnv-<Variant>-v0',
    entry_point='jlab_opt_control.envs:NewEnv',
    kwargs={...constructor defaults...},
)
```

Envs use a longer, descriptive id than agents/models/buffers — e.g.
`DnC2s-Circle2D-Statefull-v0` (lab/project prefix, env name, variant,
version). If you're adding a variant of an existing env family, match its
prefix; otherwise pick a short, descriptive prefix of your own. The
mandatory `-v<N>` suffix still applies.

No packaged cfg step here unless you added one in step 1 — skip straight to
registering.

## 3. Export

Nothing further — the import above is also the export.

## 4. Test

Add tests to `utests/test_envs_and_utils.py` following `TestCircle2DEnv`:
action/observation space shape and bounds, `reset()` returns `(obs, info)`,
`step()` returns a 5-tuple, episode termination behaves as expected
(`max_episode_steps` if you have one), and a registry-instantiation test via
`envs.make('your-id')`.

Run `bash utests/run_branch_utests.sh` before considering this done.
