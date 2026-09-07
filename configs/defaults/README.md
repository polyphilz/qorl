# Experiment defaults

Each experiment method has an independently numbered, complete TOML template.

Do not edit individual fields in any of the default configs. If changes are
required, version up the config by duplicating the file and renaming it.

Model identities are placeholders. SFT generation also leaves its generator and
generation count unset. Dataset reuse removes those generation fields.
PostgreSQL and pool definitions stay in `docker/`; experiments reference them.

One `model.context_length` supplies the agent, server, and trainer limit. SFT
batch size counts packed rows. RL batch size counts rollouts and must divide into
whole groups. Checkpoint saving does not trigger live evaluation.
