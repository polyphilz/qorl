# Model configurations

Each directory under `configs/` contains a `modelconf.json` and a short README.
The JSON defines the pinned model revision, vLLM version, connection details,
context and turn limits, and sampling settings under `policy`.

| Configuration | Context tokens | Presence penalty |
| --- | ---: | ---: |
| [000-modelconf](configs/000-modelconf/README.md) | 262,144 | 2.0 |
| [001-modelconf](configs/001-modelconf/README.md) | 20,480 | 0.0 |

All other settings are identical. Experiments select a `modelconf.json` through
their `policy_config` or `run_config` field. The served model's context length
must match the selected config; client preflight checks it.

These files contain configuration, not model weights. Base weights live in the
Hugging Face cache; trained adapters and merged models live under `outputs/`.
The random baseline has no model and keeps its policy config in experiment 000.
