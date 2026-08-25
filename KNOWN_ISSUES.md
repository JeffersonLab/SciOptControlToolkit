# Known Issues & Inconsistencies

Produced during an architecture review of `jlab_opt_control` ahead of building a
developer skill for adding new agents/models/buffers/envs. Everything below was
verified against the code on `main` (not inferred from README/docstrings alone).
Each entry lists where it lives, what it does, and its current status.

Companion reference: the full architecture/conventions writeup (registry deep
dive, driver lifecycle, config precedence, testing conventions) was produced
alongside this doc and covers the "how it works" side; this file is scoped to
"what's actually broken or inconsistent."

**Update:** B1–B5 and the I3 registry consolidation have since been
implemented on branch `7-add-agentic-skill-for-developers` (see status lines
below and the diff for details). Verified with `py_compile` and a standalone
functional check of the new `core/registry_core.py` against every original
failure mode (duplicate id, unregistered id, deprecated entry_point, missing
module, string vs. callable `entry_point`) — the project's own pytest suite
could **not** be run in the environment that made these changes, since
TensorFlow/Gymnasium/PyTorch aren't installed there. **Run
`bash utests/run_branch_utests.sh` (or the full MR suite) before merging.**

---

## 1. Confirmed bugs

These change runtime behavior, not just style.

### B1 — `raise log.error(...)` raises the wrong exception

`Logger.error(msg)` logs the message and returns `None`. `raise None` raises
`TypeError: exceptions must derive from BaseException`, discarding the
intended error message underneath a generic, unrelated traceback.

Present in all four registries, identical pattern each time:

| File | Lines | Triggered by |
|---|---|---|
| `jlab_opt_control/agents/registration.py` | 24, 61, 71, 75 | deprecated entry_point, missing module, unknown id, duplicate id |
| `jlab_opt_control/models/registration.py` | 24, 61, 71, 75 | same |
| `jlab_opt_control/buffers/registration.py` | 24, 61, 72, 76 | same |
| `jlab_opt_control/envs/registration.py` | 52, 89, 100, 104 | same |

**Status: fixed**, as part of the I3 registry consolidation — the new shared
`jlab_opt_control/core/registry_core.py` logs each message via
`self.log.error(msg)` and then raises a real exception (`ValueError` for
deprecated/duplicate ids, `ImportError` for a missing module, `KeyError` for
an unregistered id) carrying the original message text. All four
`registration.py` files now delegate to it. Verified with a standalone
functional check (all four failure modes correctly log **and** raise with the
intended message).

### B2 — `actor_gaussian-v0` is registered with the wrong packaged cfg

`jlab_opt_control/models/__init__.py` registers:

```python
register(
    id='actor_gaussian-v0',
    entry_point='jlab_opt_control.models:ActorGaussian',
    kwargs={'cfg': 'actor_fcnn.cfg'},   # should be 'actor_gaussian.cfg'
)
```

`ActorGaussian` (`models/actor_gaussian_v0.py`) requires
`len(activation_functions) == hidden_layers + 1` with the final entry `"tanh"`
(one activation per hidden layer, plus one for the output layer). The
dedicated `cfgs/actor_gaussian.cfg` is shaped correctly for this
(`["relu", "relu", "tanh"]`, 2 hidden layers). `cfgs/actor_fcnn.cfg` — what the
registry actually loads — has exactly `hidden_layers` activations ending in
`"relu"` (`["relu", "relu"]`), shaped for the *plain* `ActorFCNN`, not the
Gaussian one.

**Effect:** every `jlab_opt_control.models.make('actor_gaussian-v0', ...)` call
logs two config-mismatch errors (`act_log.error`, non-fatal) and silently
builds with the wrong architecture description. Direct instantiation
(`ActorGaussian(...)` without going through the registry) is unaffected,
since the class's own default is the correct file.

**Status: fixed.** `models/__init__.py` now registers `actor_gaussian-v0`
with `kwargs={'cfg': 'actor_gaussian.cfg'}`.

### B3 — `critic_uncertainty_fcnn-v0` cfg is wrong, missing, *and* the class's own validation check has an inverted sign

Three stacked problems on the same component, the third found only while
fixing the first two:

1. `jlab_opt_control/models/__init__.py` registered `critic_uncertainty_fcnn-v0`
   with `kwargs={'cfg': 'critic_fcnn.cfg'}` — the plain critic's cfg, not its
   own.
2. `CriticUncertaintyFCNN.__init__` (`models/critic_uncertainty_fcnn.py`)
   itself defaulted to `cfg='critic_uncertainty_fcnn.cfg'` — **a file that
   did not exist anywhere in `jlab_opt_control/cfgs/`**. Any direct
   instantiation without an explicit `cfg=` kwarg (bypassing the registry)
   would raise `FileNotFoundError` outright.
3. **(B3b, newly found)** The class's own validation check had an inverted
   sign, so *no* cfg — not even a correctly-shaped one — could ever pass it.
   Line 38 read:

   ```python
   if hidden_layers != len(nodes_per_layer) or hidden_layers != len(activation_functions)+1:
   ```

   But the constructor's own hidden-layer loop (`for i in range(hidden_layers): ... activation_functions[i]`)
   plus its use of `activation_functions[-1]` for the output layer requires
   `len(activation_functions) == hidden_layers + 1` — exactly what the
   class's *own* fallback default already assumes
   (`["relu"] * hidden_layers + ["linear"]`, line 34, which is `hidden_layers + 1`
   long). The written check instead only passed when
   `len(activation_functions) == hidden_layers - 1`, a shape that — one
   short of what the loop actually indexes — would raise `IndexError` before
   ever reaching that point. In other words: the objectively correct cfg
   shape *always* logged a spurious "activation count mismatch" error, and
   the shape the check silently accepted couldn't have run at all. Sibling
   class `ActorGaussian` has the analogous check written correctly as
   `hidden_layers != len(activation_functions)-1` — this one had the sign
   flipped, most likely a copy-paste-and-forgot-to-adjust from `CriticFCNN`'s
   simpler `hidden_layers != len(activation_functions)` check.

**Status: fixed.**
- `models/__init__.py` now registers `critic_uncertainty_fcnn-v0` with
  `kwargs={'cfg': 'critic_uncertainty_fcnn.cfg'}`.
- Added `jlab_opt_control/cfgs/critic_uncertainty_fcnn.cfg`:
  `{"hidden_layers": 2, "nodes_per_layer": [256, 256], "activation_functions": ["relu", "relu", "linear"]}`
  (2 + 1 = 3 activations, matching the class's own default shape).
- `models/critic_uncertainty_fcnn.py:38` changed `+1` to `-1`, matching
  `ActorGaussian`'s pattern and the constructor's actual indexing.
- Verified by hand against the new cfg's exact values: both the node/layer
  count check and the `activation_functions[-1] == "linear"` check now pass,
  and every index the constructor touches (`activation_functions[0]`,
  `[1]`, `[-1]`) is in range.

### B4 — stray debug `print()` statements in every registry

`load()` in all four `registration.py` files:

```python
def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    ...
```

`envs/registration.py:97` additionally has `print(self.disc_specs)` inside
`EnvRegistry.spec()`, dumping the entire env registry to stdout on *every*
`envs.make()` call, not just on error.

**Status: fixed**, as part of the I3 consolidation — the shared `load()` in
`core/registry_core.py` uses `logging.getLogger("Registry").debug(...)`
instead of `print()`, and the `print(self.disc_specs)` in `EnvRegistry.spec()`
had no equivalent carried forward (nothing else read it; it was pure debug
noise).

### B5 — (found during the I3 fix) envs registry's messages all said "agent"

`envs/registration.py` was a copy of `agents/registration.py` with the class
names renamed but several **strings left unchanged**: `EnvRegistry.make()`
logged `'Making new agent: %s'` (not "env"), and the module-not-found error
in `EnvRegistry.spec()` read `'A module ({}) was specified for the agent ...
calling `agent_module.make()`'` — every env lookup and creation was logged
under the wrong noun. Cosmetic (never affected behavior), but confusing in
logs.

**Status: fixed** automatically by the I3 consolidation — `core/registry_core.py`'s
`Registry` parametrizes every message by `self.kind` (`"Environment"` for the
env registry), so the noun is always correct.

---

## 2. Structural inconsistencies

Not wrong, but inconsistent enough to cause confusion or drift if copied
forward uncritically. Where a standard was already decided (during skill
scoping), it's noted — implementation is still pending.

### I1 — Two ways to read the same cfg dict

Agents and buffers always route reads through
`cfg_utils.cfg_get(data, key, default)` (logs an error on a missing key,
still returns the default). Models (`actor_fcnn.py`, `critic_fcnn.py`,
`actor_gaussian_v0.py`, `critic_uncertainty_fcnn.py`, …) instead call
`cfg_data.get(key, default)` directly — plain `dict.get`, silent on a miss —
despite most of them importing `cfg_utils` at the top of the file and never
using it.

**Decision:** standardize on `cfg_utils.cfg_get` everywhere, including
models. **Status:** not yet applied to existing model files.

### I2 — License header presence is inconsistent

A 27-line JLab BSD-style header appears at the top of some files and not
others, with no topical rule — it correlates with "older, 2020-era file" more
than anything else.

- **Has it:** `agents/`, `models/`, `buffers/` `__init__.py`;
  `envs/registration.py`; `circle_env.py`; `run_continuous.py`;
  `keras_td3.py`, `keras_sac.py`; `test_registry.py`.
- **Missing:** `core/*.py`; all four `registration.py`; `actor_fcnn.py`,
  `critic_fcnn.py`, `er.py`, `per.py`; `sindy_network.py`; most newer test
  files; `setup.py`.

**Decision:** don't require it going forward — treat the header as legacy,
matching what every recently-added file already does. **Status:** informational
only; no cleanup planned (existing headers are not being removed).

### I3 — The registry pattern is copy-pasted four times

`agents/registration.py`, `models/registration.py`, `buffers/registration.py`,
and `envs/registration.py` each define an independent, near-identical
`Spec`/`Registry` class pair (`AgentSpec`/`AgentRegistry`,
`ModelSpec`/`ModelRegistry`, `ReplaySpec`/`ReplayRegistry`,
`EnvSpec`/`EnvRegistry`) — same logic, different names, no shared base class.
A tell that it was copied forward rather than re-derived: the internal dict
attribute is called `disc_specs` in three of the four (agents, models, envs —
short for "discrete," from an earlier, unrelated version of this pattern)
even though none of them are about discreteness; only the buffer registry
renamed its internals to `replay_specs`.

**Decision:** consolidate into a shared `BaseSpec`/`BaseRegistry` in `core/`
before building the add-agent/model/buffer/env skill, so future registries
inherit instead of copy-pasting. Public API per module
(`register`/`make`/`spec`/`list_registered_modules`) stays unchanged.

**Status: implemented.** Added `jlab_opt_control/core/registry_core.py`
(`Spec` + `Registry`, parametrized by a `kind` string used for both the
logger name and error text). All four `registration.py` files are now thin
wrappers: a module-level `Registry("<Kind>")` instance plus four
pass-through functions. The internal instance/attribute names
(`disc_registry`/`disc_specs`, `replay_registry`/`replay_specs`,
`env_registry`) were never referenced outside each `registration.py` itself
(confirmed by repo-wide grep) — the per-module public functions
(`register`, `make`, `spec`, `list_registered_modules`) that `__init__.py`
and everything else actually imports are unchanged.

### I4 — Inconsistent naming on versioned variants

- Class `ActorFCNN_v2` (underscore) vs. registry id `actor_fcnn-v2` (hyphen,
  no separator before `v2`).
- Class `CriticFCNN_v1`/`CriticFCNN_v2` use an underscore where the unversioned
  base `CriticFCNN` doesn't.
- Agents use `PascalCase` ids (`KerasTD3-v0`); models/buffers use
  `snake_case`/`UPPERCASE` ids (`actor_fcnn-v0`, `ER-v0`) — a per-kind
  convention, not a bug, but easy to get wrong when adding a new kind.

**Status:** informational; not planned for cleanup, but the skill should not
propagate the underscore/hyphen mismatch into new versioned variants.

### I5 — Two docstring dialects

`core/*.py` abstract methods use a terse one-liner
(`""" Do a soft update of the target model """`). `drivers/run_continuous.py`
uses full Google-style `Args:`/`Returns:`/`Raises:` docstrings. Concrete
agents/models/buffers mostly have no docstrings beyond the inherited
one-liners. Looks chronological (older core vs. newer driver) rather than
deliberate.

**Status:** informational; no standard chosen yet.

### I6 — `save_cfg()` overwrite behavior differs by component kind

Models guard their `shutil.copy` with `if not os.path.exists(dest)` (copy
once, first call wins). Agents and buffers copy unconditionally on every
`save_cfg()` call (always overwrite). Both are harmless in practice, but a
new component copied from the "wrong" sibling will silently pick up the
other behavior.

**Status:** informational; no standard chosen yet.

### I7 — `run_episode()` stores `terminate` only, not `terminate or truncate`

`drivers/run_continuous.py`, in `run_episode()`:

```python
next_state, reward, terminate, truncate, _ = env.step(action)
...
if train:
    agent.memory((state, action, reward, next_state, terminate))
```

Only `terminate` is written into the replay tuple's `done` flag; a
`truncate`-only ending (e.g. hitting `TimeLimit`) is stored as `done=False`.
This is actually the technically-correct RL convention (a truncated episode
shouldn't bootstrap as if the MDP terminated) — flagged here as **worth
confirming intentional** rather than as a confirmed bug, since nothing in the
codebase states the intent explicitly.

**Status:** needs a decision — confirm intentional, or treat as a bug.

---

## 3. Not fixed in this pass

`test_baselines.py` (slow, full-training-loop tests against real Gym envs)
and `run_utests.sh` (plain-`python` test runner, not wired into CI) were
reviewed for scope but are working as intended — no action needed, just
noted here since they're easy to mistake for dead code.

---

*Last verified against `jlab_opt_control` on branch
`7-add-agentic-skill-for-developers` (tip `48e624f`, same as `main` at review
time).*
