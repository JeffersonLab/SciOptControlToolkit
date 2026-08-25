# Recipe: new replay buffer

Assumes you've read `conventions.md`.

## 1. Implement

`buffers/new_buffer.py`, subclassing `jlab_opt_control.core.Replay`.
Required: `record(memory)`, `sample(nsamples)`, `save(filename)`,
`save_cfg()`, `load(filename)`, `size()`.

Constructor signature convention (matches `ER`/`PER`):

```python
def __init__(self, state_dim, action_dim, logdir, buffer_size=None, cfg='new_buffer.cfg'):
```

If you're adding a *prioritized* variant, consider subclassing `ER` the way
`PER` does, rather than reimplementing the circular-buffer bookkeeping from
scratch — `ER` already handles the pointer/capacity/save-load logic
correctly, including the pointer round-tripping through save/load (a past
bug: an old heuristic guessed the write pointer from which rows were
non-zero, which silently lost trailing all-zero experiences; `ER` now saves
the pointer explicitly — don't reintroduce a heuristic-based pointer).

**Read this before assuming your prioritized buffer will just work with
existing agents.** Every TD3/DDPG/SAC agent's `train()` calls
`self.buffer.update_priorities(...)` gated on a **string substring check**,
not an interface/type check:

```python
if "PER" in self.buffer_type:
    new_priorities = td_errors.numpy().squeeze()
    self.buffer.update_priorities(new_priorities)
```

If your new prioritized buffer's registered id doesn't literally contain the
substring `"PER"`, no existing agent will ever call `update_priorities` on
it — it'll silently behave like uniform sampling no matter how you implement
`sample()`. Either name your id to include `PER` (e.g. `RankPER-v0`), or be
aware you're also on the hook for updating this check in every agent that
should support it, which is a much bigger, more invasive change — flag that
tradeoff to the user rather than silently picking one.

## 2. Configure

Add `cfgs/new_buffer.cfg` with your buffer's hyperparameters (e.g.
`buffer_capacity`, and for prioritized variants: `alpha`, `beta`,
`beta_increment`, `prioritization_type`).

## 3. Register

In `buffers/__init__.py`:

```python
from jlab_opt_control.buffers.new_buffer import NewBuffer
...
register(
    id='NewBuffer-v0',
    entry_point='jlab_opt_control.buffers:NewBuffer',
    kwargs={'cfg': 'new_buffer.cfg'},
)
```

Then: `python .claude/skills/soct-add-component/scripts/check_cfg_shape.py buffer NewBuffer-v0`.

## 4. Export

Nothing further — the import above is also the export.

## 5. Test

No shared mixin for buffers — add a standalone `TestCase` in
`utests/test_buffers.py` following `TestERBuffer`/`TestPERBuffer`: size
tracking (starts at zero, increments, caps at capacity), record/sample
shapes, circular overwrite, save/load round-trip (including the pointer —
see the note above), `save_cfg`, and a registry-instantiation test.

Run `bash utests/run_branch_utests.sh` before considering this done.
