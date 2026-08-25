---
name: soct-add-component
description: >
  Add or scaffold a new component in this repo's jlab_opt_control RL framework
  (SciOptControlToolkit): a new agent (TD3/DDPG/SAC/SINDy/etc. variant), a new
  model (actor/critic/SINDy network architecture), a new replay buffer, a new
  Gymnasium environment, or a new driver/training workflow. Use this any time
  someone asks to add, create, implement, register, or wire up one of these —
  even when they don't use those exact words: "I want to try a CrossQ agent",
  "add a ring-buffer replay variant", "we need an env for the new beamline",
  "write a sweep script over these agents". Always consult this before
  hand-writing such code in this repo: the codebase enforces a specific
  constructor signature, cfg-override precedence, registration pattern, and
  test-mixin contract per component kind that is easy to get subtly wrong —
  this package has already shipped two silent bugs from exactly that (see
  KNOWN_ISSUES.md B2/B3). Not for general RL/Gymnasium/TensorFlow questions
  unrelated to extending this specific package.
---

# Adding a component to jlab_opt_control

Every extension point in this package — agent, model, buffer, env, driver —
follows the same shape: **implement → configure → register → export → test.**
The five kinds differ only in base class, folder, and a few conventions; the
scaffolding around them (registry, cfg precedence, testing obligations) is
shared. Read this file first, then jump to the reference for the kind you're
adding.

| Adding a... | Base class | Folder | Reference |
|---|---|---|---|
| Agent (RL algorithm) | `jlab_opt_control.Agent` | `agents/` | `references/agent.md` |
| Model (actor/critic/network) | `jlab_opt_control.core.Model` | `models/` | `references/model.md` |
| Replay buffer | `jlab_opt_control.core.Replay` | `buffers/` | `references/buffer.md` |
| Environment | `gymnasium.Env` | `envs/` | `references/env.md` |
| Driver (training/workflow script) | none — a runnable module | `drivers/` | `references/driver.md` |

For the underlying mechanics referenced from every recipe — the registry,
cfg-override precedence, logging/error conventions — see
`references/conventions.md`. It's worth reading once; every recipe assumes it.

## Why this matters more than it looks

Nothing in this codebase's abstract base classes enforces the actual
contract. `Agent.__init__` does nothing; `Model`/`Replay` don't check
constructor signatures either. The real contract — the exact constructor
shape, how cfg overrides resolve, what the registry expects, what the test
suite assumes — lives entirely in convention across the existing
implementations, and is only caught by the test suite, if at all. Deviating
from it doesn't raise an error; it produces a component that *looks* like it
works and fails quietly later (wrong architecture silently loaded, a
checkpoint that never saves, a buffer whose priorities never update). Treat
the patterns below as load-bearing, not stylistic.

## The one habit that would have prevented this package's worst bugs

Twice, a new model was registered pointing at the wrong packaged `.cfg` file
— once at a sibling model's cfg, once at a cfg that didn't exist at all (see
`KNOWN_ISSUES.md`, B2/B3). Neither raised an exception: this codebase's cfg
validation logs an error and keeps going (see `references/conventions.md`),
so both shipped silently and were only caught by manual review. **After
registering any agent, model, or buffer, run:**

```bash
python .claude/skills/soct-add-component/scripts/check_cfg_shape.py <kind> <registry-id>
```

(`<kind>` is `agent`, `model`, or `buffer`.) It constructs the component
through the real registry with a throwaway logdir and dummy dimensions, and
loudly surfaces any `ERROR`-level log line raised during construction —
exactly the errors this codebase would otherwise swallow. It needs the
project's conda env active (`conda activate jlab_opt_control_env`), since it
imports TensorFlow/Gymnasium through the real package.

## Shared checklist (applies to every kind)

1. **Implement** the class, subclassing the right base, in the right folder.
2. **Configure**: add a packaged `cfgs/<name>.cfg` (JSON) if your kind uses
   one (agents, models, and buffers do; environments usually don't).
3. **Register**: add the import + `register(id=..., entry_point=..., kwargs={...})`
   call in that folder's `__init__.py`. Double-check `kwargs['cfg']` points at
   *your* cfg file, not a sibling's.
4. **Export**: nothing extra to do — the import in step 3 is also the export.
5. **Test**: add the mixin subclass / test class the existing suite expects
   for your kind (see the per-kind reference and
   `references/conventions.md`'s testing section). This is not optional
   busywork — `test_registry.py`'s blanket smoke test only proves your
   component *constructs*; it proves nothing about correctness, and nothing
   forces you to add real coverage beyond it.
6. **Run the cfg check** (above) if your kind has a cfg file.
7. **Run the tests**: `bash utests/run_branch_utests.sh` for the fast suite;
   see `references/conventions.md` for when to also run the MR suite or
   `test_baselines.py`.

If you're adding something that doesn't cleanly fit one kind (e.g. a model
that's also somewhat env-specific), still pick the closest existing kind and
follow its recipe — this package has no precedent for a sixth category, and
inventing one is a bigger decision than this skill should make for you. Flag
it to the user instead of guessing.
