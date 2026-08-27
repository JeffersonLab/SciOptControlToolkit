# Shared conventions

Read once; every recipe (agent/model/buffer/env/driver) assumes this.

## The registry

Agents, models, buffers, and envs are each looked up by a short string id
through `jlab_opt_control.core.registry_core.Registry` — one shared
implementation (as of the registry-consolidation fix; each kind's
`registration.py` is now a ~15-line wrapper exposing `register`, `make`,
`spec`, `list_registered_modules` for that kind).

```python
# <kind>/registration.py — you shouldn't need to touch this file
from jlab_opt_control.core.registry_core import Registry
agent_registry = Registry("Agent")
def register(id, **kwargs): return agent_registry.register(id, **kwargs)
def make(id, **kwargs): return agent_registry.make(id, **kwargs)
...
```

What you actually write is one `register(...)` call in `<kind>/__init__.py`:

```python
register(
    id="KerasX-v0",
    entry_point="jlab_opt_control.agents:KerasX",
    kwargs={"cfg": "keras_x.cfg"},   # defaults merged with, then overridden by, call-site kwargs
)
```

`make(id, **kwargs)` merges `kwargs` from the `register()` call with whatever
is passed at call time — **call-site kwargs always win**. This is how the
driver's `--actor_cfg`, `--buffer_type`, etc. flags override a component's
packaged defaults without needing to touch the registration itself.

**ID convention:** always end in `-v<N>` (mandatory even for the first
version, e.g. `-v0`). Agents use `PascalCase` (`KerasTD3-v0`); models and
buffers use `snake_case`/`UPPERCASE` matching their acronym
(`actor_fcnn-v0`, `ER-v0`); envs use a longer descriptive prefix
(`DnC2s-Circle2D-Statefull-v0`). Match whichever family your kind already
uses — don't invent a new casing style.

**Registering an id twice raises** (`ValueError: Cannot re-register id: ...`).
Making an unregistered id raises `KeyError`. Both are logged *and* raised
with the real message — if you're ever tempted to write
`raise some_logger.error(msg)` anywhere in this codebase, don't:
`Logger.error()` returns `None`, and `raise None` raises an unrelated
`TypeError` that discards your message. This was a real, since-fixed bug in
every registry (see `KNOWN_ISSUES.md` B1) — don't reintroduce the pattern.

## Config files: resolution, precedence, and the bug this package already hit twice

`cfgs/*.cfg` files are plain JSON (the `.cfg` extension is misleading).
Every agent/model/buffer resolves its packaged cfg the same way, relative to
its *own* defining file:

```python
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, "../cfgs/")
self.pfn_json_file = os.path.join(full_path, cfg)   # cfg = the constructor's cfg= argument
with open(self.pfn_json_file) as f:
    data = json.load(f)
```

**Read values through `cfg_utils.cfg_get(data, key, default)`**, not bare
`dict.get(key, default)`. Functionally they're close (both return `default`
on a miss), but `cfg_get` logs an error when the key is missing, which is the
difference between noticing a typo'd key immediately and finding out weeks
later. (Some existing model files still use bare `dict.get` — that predates
this being settled as the standard; write new code against `cfg_get`.)

**Precedence, highest to lowest:**

1. A kwarg passed directly to the constructor (or forwarded from the CLI,
   e.g. `--actor_cfg /path/custom.cfg`) — forwarded to `agents.make()` etc.
   only when not `None`, so an *unset* CLI flag never shadows a real default
   with a literal `None`.
2. The packaged default named in `register(..., kwargs={"cfg": "..."})`.
3. A hardcoded fallback inside the constructor itself, via
   `cfg_utils.cfg_get(data, key, some_literal_default)`.

**The bug to not repeat:** a model's `__init__` validates its cfg shape
(e.g. "does `activation_functions` have the right length for this many
hidden layers?") but only *logs an error* on mismatch — it never raises.
Combined with each model expecting a *different* activation-count convention
(see `model.md`), it's easy to register a real, existing,
validly-parsing cfg file that is simply the *wrong shape for this class*,
and nothing will stop it from running with a silently-broken architecture.
This shipped twice in this codebase (`KNOWN_ISSUES.md` B2 and B3) before
being caught by manual review. Always run
`scripts/check_cfg_shape.py <kind> <id>` after registering — it constructs
the real thing and surfaces any `ERROR`-level log line that construction
would otherwise swallow.

**`save_cfg()`**: every agent/model/buffer copies its *resolved* cfg file
into `<logdir>/cfgs/<basename>` at construction time, via `shutil.copy`
wrapped in a bare `try/except` (never raises). Models guard with
`if not os.path.exists(dest)` (copy once); agents/buffers overwrite
unconditionally. Either is fine — match whichever your sibling
implementations of the same kind do.

## Logging & error-handling policy

Every module declares its own logger and re-calls `logging.basicConfig`:

```python
my_log = logging.getLogger("MyComponent")
my_log.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
```

`basicConfig` is a no-op after the first call per process, so this is
harmless to repeat, but don't rely on your file's format string actually
taking effect — whichever module happens to import first wins.

Two deliberate, distinct policies coexist — match the one appropriate to what
you're writing, don't default to one everywhere:

- **I/O convenience methods** (`save`, `load`, `save_cfg`): wrap the body in
  a bare `except:` and log-and-continue. A failed checkpoint write should
  never crash a training run.
- **Construction-time validation** (e.g. an invalid action space): log an
  error and `sys.exit(0)` — fail fast, since nothing downstream can recover
  from a malformed agent.

## License header

Not required on new files. Several older files carry a JLab BSD-style
copyright block, but every recently-added file omits it — treat that as the
current standard, not something to restore.

## Testing conventions

Tests are `unittest.TestCase` classes, runnable standalone
(`python utests/test_x.py`) but run through `pytest` in CI.

**Agents use a shared-behavior mixin** — this is the pattern most likely to
bite you if skipped. `utests/test_agents.py` defines `AgentTestMixin` (action
shape/bounds, memory, train, soft_update, save/load) and
`utests/test_cli_kwargs.py` defines `AgentKwargsFallbackMixin` (constructor
signature shape, kwarg-over-cfg precedence for every overridable setting).
Adding a new agent means adding one short subclass to each:

```python
class TestKerasXAgent(AgentTestMixin, unittest.TestCase):
    agent_cls = KerasX
    # override only the specific tests that genuinely differ for this agent
```

Models and buffers currently do **not** have an equivalent mixin — each gets
its own standalone `TestCase` in `test_models.py` / `test_buffers.py`,
covering output shape/bounds, `save_cfg`, and a registry-instantiation test.
Follow that (no-mixin) pattern for consistency with the existing sibling
tests, rather than introducing a mixin unprompted.

**The free registry smoke test**: `test_registry.py` iterates
`list_registered_modules()` for each kind and instantiates every id with
minimal, standard kwargs (agents: `env=gym.make('MountainCarContinuous-v0')`,
`logdir='./'`). Registering a new id gets this for free — but it only proves
construction succeeds, not that the cfg shape or algorithm logic is correct.

**Three ways to run tests, three different purposes:**

| Script | Scope | Use it for |
|---|---|---|
| `bash utests/run_branch_utests.sh` | agents, buffers, envs_and_utils, models, registry | Your everyday check while developing — this is what runs on every branch push in CI. |
| `bash utests/run_mr_utests.sh` | the entire `utests/` directory, with JUnit output | Before opening/updating a merge request — this is the full CI gate. |
| `bash utests/run_utests.sh` | every `test_*.py`, via plain `python file.py` (no pytest) | A dependency-light fallback if pytest itself is unavailable. |

`utests/test_baselines.py` is a separate, slow suite of full training runs
against real Gym environments (e.g. `HalfCheetah-v4` for up to 10000
episodes) — not part of either automated script above. Run it by hand
(`python utests/test_baselines.py`, or add a case following its existing
pattern) when you want end-to-end confidence that a new agent actually
learns, not just that it constructs and doesn't crash.
