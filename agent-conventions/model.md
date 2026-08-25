# Recipe: new model (actor / critic / network)

A model is a `tf.keras.Model` — an actor, a critic, or a network like the
SINDy ones. Assumes you've read `conventions.md`.

## 1. Implement

`models/x_name.py`, subclassing `jlab_opt_control.core.Model`
(itself a `tf.keras.Model`). Required: `call(self, ..., training=False)`,
`save_cfg()`.

Most models (the FCNN family) share a dynamic-architecture pattern driven by
three cfg keys: `hidden_layers`, `nodes_per_layer` (list, one entry per
hidden layer), `activation_functions`. **The length `activation_functions`
must be is not the same across every model** — it depends on whether the
class needs a dedicated final-layer activation distinct from its hidden
layers. Get this wrong and the mismatch is only ever *logged*, never raised
(see `conventions.md`) — this table exists because getting it wrong shipped
twice:

| Class | `len(activation_functions)` | Last entry must be | Why |
|---|---|---|---|
| `ActorFCNN` | `hidden_layers` | (output hardcoded to `"tanh"`, not from cfg) | Output layer's activation isn't cfg-driven at all. |
| `ActorGaussian` | `hidden_layers + 1` | `"tanh"` | One activation per hidden layer, plus one dedicated entry (`[-1]`) checked against the mean/std output layers. |
| `CriticFCNN` | `hidden_layers` | (output hardcoded to `"linear"`) | Same as ActorFCNN — output isn't cfg-driven. |
| `CriticUncertaintyFCNN` | `hidden_layers + 1` | `"linear"` | Same shape as ActorGaussian — dedicated `[-1]` entry feeds the (mean, variance) output layer, and it must be linear, not squashed. |

**When you add a new model, decide which of these two shapes it follows
*before* writing its cfg** — does your output layer have its own dedicated
activation driven by `activation_functions[-1]` (Gaussian/Uncertainty shape,
`hidden_layers + 1` entries), or is the output activation hardcoded in `call()`
(FCNN shape, exactly `hidden_layers` entries)? Whichever it is, write the
`__init__` validation check to match — and double check the direction of the
comparison; `CriticUncertaintyFCNN` shipped with `+1` where the code needed
`-1` for months, meaning *no* valid cfg could ever pass its own check (see
`KNOWN_ISSUES.md` B3b).

Skeleton for the FCNN-style shape:

```python
class NewModel(Model):
    def __init__(self, state_dim, action_dim, logdir, cfg='new_model.cfg'):
        super().__init__()
        # ... resolve self.pfn_json_file exactly as in conventions.md ...
        with open(self.pfn_json_file, 'r') as f:
            cfg_data = json.load(f)
        hidden_layers = cfg_utils.cfg_get(cfg_data, 'hidden_layers', 2)
        nodes_per_layer = cfg_utils.cfg_get(cfg_data, 'nodes_per_layer', [256, 256])
        activation_functions = cfg_utils.cfg_get(cfg_data, 'activation_functions', ["relu"] * hidden_layers)
        self.logdir = logdir

        if hidden_layers != len(nodes_per_layer):
            my_log.error("Number of nodes per layer does not match hidden_layers.")
        # ... your activation_functions length check, matching the table above ...

        self.hidden_layers = [
            layers.Dense(nodes_per_layer[i], activation=activation_functions[i],
                         input_shape=(state_dim,) if i == 0 else ())
            for i in range(hidden_layers)
        ]
        self.output_layer = layers.Dense(..., activation=...)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def save_cfg(self):
        # copy self.pfn_json_file to <logdir>/cfgs/, guarded by `if not os.path.exists(dest)`
        ...
```

Use `cfg_utils.cfg_get` (not bare `dict.get`) for new model code — see
`conventions.md`.

## 2. Configure

Add `cfgs/new_model.cfg`, shaped per the table above. Don't reuse an
existing sibling's cfg file "because it's close enough" — that's exactly how
B2 and B3 happened.

## 3. Register

In `models/__init__.py`:

```python
from jlab_opt_control.models.new_model import NewModel
...
register(
    id='new_model-v0',
    entry_point='jlab_opt_control.models:NewModel',
    kwargs={'cfg': 'new_model.cfg'},   # your own file — verify this by eye before moving on
)
```

Then: `python agent-conventions/scripts/check_cfg_shape.py model new_model-v0`
— this is the step that would have caught both B2 and B3 immediately instead
of on manual review.

## 4. Export

Nothing further — the import above is also the export.

## 5. Test

Models don't use a shared mixin (unlike agents) — add a standalone
`TestCase` in `utests/test_models.py` following the existing pattern
(`TestActorFCNN`, `TestCriticFCNN`): output shape, output within bounds (for
actors), `training=True/False` doesn't crash, `save_cfg` writes the file,
and a registry-instantiation test:

```python
class TestNewModel(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.model = NewModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, logdir=self.logdir)

    def test_output_shape(self): ...
    def test_save_cfg(self): ...
    def test_registry(self):
        m = models.make('new_model-v0', state_dim=STATE_DIM, action_dim=ACTION_DIM, logdir=self.logdir)
        self.assertIsInstance(m, NewModel)
```

Run `bash utests/run_branch_utests.sh` before considering this done.
