# Recipe: new agent

An agent is an RL algorithm (TD3, DDPG, SAC, and this codebase's SINDy/
uncertainty variants of them). Assumes you've read `conventions.md`.

## 1. Implement

`agents/keras_x.py`, subclassing `jlab_opt_control.Agent`
(`core/agent_core.py`). Required abstract methods:
`soft_update`, `train`, `action`, `load`, `save`, `save_cfg`.

**Also implement `memory(self, obs_tuple)`, even though it isn't in the
ABC.** `drivers/run_continuous.py`'s `run_episode()` calls
`agent.memory((state, action, reward, next_state, terminate))` on every
training step, unconditionally. It's a de facto required method that the
abstract base class doesn't enforce — miss it and the agent constructs fine,
passes the registry smoke test, and only fails the first time it's actually
run through the driver or `AgentTestMixin`.

**Constructor signature — exact, and tested for exactly this shape**
(`test_cli_kwargs.py`'s `test_signature_is_env_logdir_cfg_kwargs`):

```python
def __init__(self, env, logdir, cfg='keras_x.cfg', **kwargs):
```

Resolve every overridable setting the same way — kwarg wins, else the
packaged cfg, via `cfg_utils.cfg_get`:

```python
self.buffer_type    = kwargs.get('buffer_type', cfg_utils.cfg_get(data, 'buffer_type', None))
buffer_size         = kwargs.get('buffer_size', cfg_utils.cfg_get(data, 'buffer_size', None))
self.model_load_path = kwargs.get('load_model', cfg_utils.cfg_get(data, 'load_model', None))
self.actor_cfg      = kwargs.get('actor_cfg')
self.critic_cfg     = kwargs.get('critic_cfg')
self.buffer_cfg     = kwargs.get('buffer_cfg')
```

Build sub-components through *their* registries, not by importing classes
directly — this is what lets `--actor_cfg`/`--critic_cfg`/`--btype` swap
architectures without touching agent code:

```python
self.buffer = jlab_opt_control.buffers.make(self.buffer_type, state_dim=..., action_dim=..., logdir=self.logdir, buffer_size=buffer_size, **({"cfg": self.buffer_cfg} if self.buffer_cfg else {}))
self.actor_model = jlab_opt_control.models.make(self.actor_model_type, state_dim=..., action_dim=..., min_action=..., max_action=..., logdir=self.logdir, **({"cfg": self.actor_cfg} if self.actor_cfg else {}))
```

Validate the action space at construction and fail fast if it's wrong
(this codebase's convention — see `conventions.md`'s error-handling section):

```python
try:
    assert "Box" in str(type(env.action_space)), 'Invalid action space'
    ...
except:
    my_log.error('Action space not valid for this agent.')
    sys.exit(0)
```

For a full worked example, read `agents/keras_ddpg.py` (simplest — single
critic, no entropy term) before `agents/keras_td3.py` (twin critics) or
`agents/keras_sac.py` (entropy tuning). Don't copy `agents/keras_td3.py`
line-for-line as a starting point unless your algorithm is actually a TD3
variant — copying an unrelated algorithm's training math and only swapping
the name is a worse starting point than writing `train`/`action` from your
algorithm's actual update rule.

## 2. Configure

Add `cfgs/keras_x.cfg` — hyperparameters plus which registered model/buffer
ids this agent uses by default:

```json
{
    "warmup_size": "2500",
    "batch_size": "256",
    "actor_learning_rate": "0.0003",
    "critic_learning_rate": "0.0003",
    "buffer_type": "ER-v0",
    "actor_model": "actor_fcnn-v0",
    "critic_model": "critic_fcnn-v0"
}
```

## 3. Register

In `agents/__init__.py`:

```python
from jlab_opt_control.agents.keras_x import KerasX
...
register(
    id="KerasX-v0",
    entry_point="jlab_opt_control.agents:KerasX",
    kwargs={"cfg": "keras_x.cfg"},
)
```

Then: `python agent-conventions/scripts/check_cfg_shape.py agent KerasX-v0`.

## 4. Export

Nothing further — the import above is also the export.

## 5. Test

In `utests/test_agents.py`:

```python
class TestKerasXAgent(AgentTestMixin, unittest.TestCase):
    agent_cls = KerasX
```

Only override a test from `AgentTestMixin` when your agent's structure
genuinely differs — e.g. SAC overrides `test_soft_update_changes_target_weights`
because it has no `target_actor`. Don't override tests just because they're
inconvenient; a failing inherited test is telling you something real.

In `utests/test_cli_kwargs.py`:

```python
class TestKerasXKwargsFallback(AgentKwargsFallbackMixin, unittest.TestCase):
    agent_cls = KerasX
    cfg_buffer_type = 'ER-v0'          # match your cfg's buffer_type
    has_second_critic = True           # False if single-critic like DDPG
```

If your agent's default actor/critic differs from the FCNN defaults (e.g. it
uses `actor_gaussian-v0`), also override `actor_cfg_data`/its counterpart —
see `TestKerasSACKwargsFallback` for the pattern.

Run `bash utests/run_branch_utests.sh` before considering this done.

## When an agent genuinely has no critic and no buffer

Not every agent fits the critic + replay-buffer shape above. `KerasDEPO`
(`agents/keras_depo.py`, DEPO = Differentiable Environment Policy
Optimization) is a critic-free, gradient-based agent: it
requires a *differentiable* environment (TF ops all the way through
`step()`, see `envs/diff_circle_env.py`), unrolls the actor forward through
it for a configurable number of steps inside one `tf.GradientTape`, and
backpropagates the discounted, done-masked sum of rewards straight into the
actor — no critic, no bootstrapped value, no replay buffer, no target
networks.

This is a deliberate, sanctioned exception, not a shortcut — follow the same
principle it followed, not its literal shape:

- Constructor signature is still exactly `(self, env, logdir, cfg=..., **kwargs)`.
- Every abstract method on `Agent` still needs a real implementation, even
  if some are one-line no-ops (`soft_update()` returns immediately; there's
  nothing to soft-update without a target network). Say why in the
  docstring/comment, don't just leave it empty and unexplained.
- `memory()` still needs to exist even though it's a no-op — the driver
  calls it unconditionally every training step regardless of which agent is
  behind it (see `conventions.md`'s registry section on why this matters).
- Don't force `buffer_type`/`buffer_size` kwargs onto an agent that has no
  use for them just to match `AgentKwargsFallbackMixin`'s shape. Skip the
  mixin, write standalone tests instead, and say in a comment why the mixin
  doesn't apply — that's what keeps this a documented exception instead of
  a silent inconsistency the next person has to re-discover.
- If your agent needs something fundamentally different from a standard Gym
  env (here: differentiability), exclude it explicitly from any blanket
  cross-agent smoke test (see `utests/test_registry.py`'s
  `test_continuous_agents`) rather than letting it fail there — with a
  comment explaining why, and a pointer to where it *is* actually tested.
