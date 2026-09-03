# Maintenance scripts

This directory contains the CEB and JOB data-preparation pipelines, Docker
checks, and the model-server launcher. Reusable runtime, evaluation, and SFT
code lives under `src/qorl/`; GPU-dependent adapter and rendering tools live
under `training/src/qorl_training/`; experiment-specific commands live with
their inputs under `experiments/`.

Run Python data commands from the repository root with
`uv run python -m scripts.<area>.<command>`.
