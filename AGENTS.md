# AGENTS.md

QORL trains a 4B language model to steer PostgreSQL's optimizer with `pg_hint_plan`
hints, using Prime-RL.

Every change must be minimal, correct and readable. Elegance is a requirement, not
a nicety. If a change makes the code larger and harder to read in order to satisfy
some constraint, say so instead of shipping it.

## Repository map

Three homes for non-code content:

- `data/` holds workloads (JOB, CEB) and their inventories. Inventories bind to data
  identity only: fixture id, snapshot id, archive checksum, PostgreSQL system
  identifier. They never record how a run was executed.
- `experiments/NNN-name/` holds one experiment's inputs and its own run script, with a
  chronological prefix. An experiment may read another experiment's frozen inputs
  read-only, and both READMEs must name that dependency. Nothing under `experiments/`
  is imported by anything.
- `configs/` holds shared configuration: policy sampling configs under
  `configs/policy/`, PostgreSQL runtime profiles under `configs/postgres/`.

Code:

- `src/qorl/` is the library. It depends on the standard library and Pydantic only.
  GPU, torch, Prime-RL, and verifiers code lives in `training/src/qorl_training/`,
  which depends on `qorl`; `qorl` never imports `qorl_training`.
- `scripts/` holds data pipelines and shell entrypoints. It defines nothing importable
  by `src/`.
- `docker/postgres/contract/` is the pinned PostgreSQL configuration. The current
  contract id is `benchmark-v2`; `geqo = off` is deliberate and documented there.

Dependency direction inside `src/qorl`, never upward:

```
util <- db <- workload <- plans <- measure <- agent <- adapters <- evaluation, sft
```

Modules are snake_case, never hyphenated, never version-suffixed. A file named `X.py`
is tested by `tests/.../test_X.py`, and `tests/qorl/` mirrors `src/qorl/`.

## Commands

Run from the repository root.

```bash
uv run pytest                      # All tests must be green before any report
uv run ruff check . && uv run ruff format --check .
uv run pyright                     # strict; see the baseline rule under Types
uv run python -m scripts.ceb.extract_sql_from_qreps --check
uv run python -m scripts.ceb.build_ceb_task_inventory --check
uv run python -m scripts.job.build_job_task_inventory --check
uv run python experiments/004-rl-run-v2/build_inventory.py --check
uv run python experiments/004-rl-run-v2/build_timeouts.py --check
```

Training-side code only runs on the Linux CUDA host:

```bash
uv run --project training --frozen python -m unittest \
  training/tests/test_adapters.py training/tests/test_runtime.py \
  training/tests/test_taskset.py
```

When work touches `training/`, say plainly whether it was run on the host or only
compiled locally.

## Types and records

- New code does not introduce `dict[str, Any]` for a domain record. A task, action,
  candidate, outcome, measurement, observation, trace, report, or manifest is a
  Pydantic model with the field names it has on disk. Plain frozen dataclasses are for
  in-memory values that are never serialized.
- One record system. Do not add `TypedDict`, `NamedTuple`, or hand-built dict
  builders beside the models. Do not cast `json.loads` output to a type; validate it.
- `object` is acceptable only at an untrusted input boundary and must be narrowed or
  validated on the next line. It never appears in an internal API.
- Do not remove `Any` mechanically. Replacing `Any` with `object` across a package is
  a useless operation. Replace it by introducing the model that owns the record.
- Strict Pyright reports 612 diagnostics at the time of writing. The count goes down
  or stays; a change that raises it is not done. Do not weaken strict mode, add blanket
  ignores, or ban `Any` in Ruff before the schemas exist.
- Enumerations are `StrEnum`s whose values are the wire strings. Comparisons use the
  enum; serialization writes `.value`.
- Interfaces are `typing.Protocol`s named for what they do (`QueryExecutor`,
  `InspectionExecutor`, `SqlSource`). `QueryExecutor` is the only thing in the code
  called a protocol; `MeasurementProtocol` is the timing procedure and keeps its name.
- No `cast()` to satisfy a type you could state. A `cast` is a `type: ignore` with
  better manners.

## Model-visible bytes and stored records

- The system prompt, tool schema, initial observation, and tool-result bytes are
  frozen from the 027 re-baseline onward. Changing any of them is a versioned protocol
  change: it is declared in a plan with a dated decision and requires re-pinned
  goldens. It should never be a side effect of a refactor.
- Goldens pin the contract: the request bytes in `tests/qorl/agent/test_agent.py`,
  the tool schema and the malformed-action feedback corpus under
  `tests/qorl/plans/fixtures/`, and the full rollout record in
  `tests/qorl/measure/golden_rollout.json`. A golden failure is a finding to report,
  not a fixture to regenerate.
- Error text the model reads is part of the contract. It names the path of the bad
  field first and never leaks a Python or library class name.
- Stored JSON field names are not renamed without a schema-version bump and a reader
  for the old version. Additive fields are allowed and are recorded in the plan.
- Numeric behavior is pinned: reward and penalties, score clipping, fingerprint
  canonicalization, the timeout formula, and execution counts. Tests assert the counts
  on actual calls.
- `Shared Read Blocks` counts buffer-pool misses served mostly from the page cache. It
  does not measure disk I/O and must not be used as a cold-cache detector.

## Numbers and settings

- No bare numeric literal outside 0, 1, and indices. Every number is a named constant
  in the module that owns the concept, or a field in a config or contract file with
  recorded provenance. If you cannot say what a number means, that is the finding.
- Policy configs are versioned files. Never edit a policy config that has a version
  less than the current one. The client preflight requires the served model's context
  length to equal the policy config's; change both or neither.
- The random sampler carries `SAMPLER_VERSION`. Any change to its action space or
  draw order bumps it, and the frozen baseline that used it is re-run.
- The PlanAction allowlist is referenced by the schema, the validator, the random
  sampler, the SFT steering families, and the schema golden. Prune or extend all of
  them together.

## Measurement

- All measurement runs on the four-worker pool profile
  `configs/postgres/training-pool-v1.json`: calibration, training, checkpoint
  and paired and live validation, the JOB benchmark, SFT generation. The single
  evaluation-worker profile exists for fixture build, restore, and smoke tests only.
- Calibration and timeout records are bound to a contract id. Records from
  `benchmark-v1` are not reused under `benchmark-v2`. New inventories are calibrated
  before anything is measured against them.
- The pool's measured run-to-run noise floor is about 2.5 percent of execution time.
  Keep it in mind when judging whether a difference is real.

## Experiments

- A frozen experiment's `run.py` is a record of how the run happened. It is edited
  only to follow a library rename or a documented protocol change, and its README says
  what would happen if it were re-run today.
- Every experiment names its data identity, its inputs, its outputs directory, and any
  read-only dependency on another experiment.
- Do not re-run frozen baselines until the settings they depend on are frozen, and
  never mix results from different contract ids or sampler versions in one table.

## Tests

- Native pytest with plain `assert`; shared fixtures live in `tests/conftest.py`,
  including the repository root and the experiment-script loader.
- Fakes are small and local. A fake satisfies a one-method Protocol or is a real
  object built from checked-in data. Never subclass a real class with a no-op
  constructor to satisfy a type.
- Assert on typed attributes, not dict keys, for records that have models.
- Frozen experiment invariants (cohort sizes, seeds, cadence) live in
  `tests/experiments/`, not in harness code. Harness loaders check sanity only.
- A corpus of cases is `parametrize`d so every failing case is reported.

## Writing code and documentation

- No forward-looking notes in code or docs: no "the future X will", "for now",
  "should later", "eventually", `TODO`. Undone work goes in a plan with a named gate,
  or in a failing or skipped test that names the gap. This rule exists because such a
  note once hid a bug that disabled join-order hints on a quarter of training tasks.
- No personal machine names in any file. Refer to "the benchmark host".
- No backward-compatibility shims, re-exports for old paths, or dead parameters.
  Delete the old thing and update its callers.
- No abstraction beyond what the plan lists. If a phase suggests one, write it down in
  the plan and continue.
- A module that shrinks the problem is better than one that wraps it. If satisfying a
  rule needs an adapter that reverse-engineers a library's output, the rule and the
  library disagree; raise it.

## Commits

- Mechanical changes (formatting, safe lint fixes, pure moves) go in their own commit
  and are listed in `.git-blame-ignore-revs`. Semantic edits, however small, go in a
  separate commit.
- A commit message says what changed and why in one line. Verification claims in a
  report must be true as stated: name what ran, where, and what did not.
