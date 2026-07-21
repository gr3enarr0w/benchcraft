# Decision: shared experiment-tracking backend (MLflow/W&B) in `dscraft.core`?

- **Status:** Decided — defer. No subpackage has a real need yet.
- **Date:** 2026-07-21
- **Issue:** [gr3enarr0w/dscraft#35](https://github.com/gr3enarr0w/dscraft/issues/35)

## Summary

Issue #35 asks whether `dscraft.core` should expose a pluggable
experiment-tracking integration point (MLflow and/or W&B) for
training-loop-having subpackages. **Recommendation: defer, do not build.**
No `dscraft` subpackage today has a real, iterative, multi-run training
loop that produces metrics worth tracking and comparing over time — the
one closest candidate (`forecast.backtest()`) is a single evaluation call
returning a plain report object, not a training loop. CLAUDE.md's "core
stays thin, build shared infra only once two real modules need it" rule
isn't met at zero real modules, let alone two. Separately, and just as
importantly: OpenTelemetry (already `dscraft.core`'s locked telemetry
schema) covers a *different* problem than experiment tracking, so this
would be a genuinely new, complementary capability if built — not a
duplicate of `dscraft.core.telemetry` — but that only matters once there's
a real subpackage need to build it against.

## (a) Which subpackages currently produce metrics/results, and how do they surface them today?

Checked every subpackage the issue names, by reading the actual code:

- **`forecast`** (`packages/dscraft/src/dscraft/forecast/backtest.py`):
  `backtest()` returns a `BacktestReport` dataclass — `metrics: list[SeriesMetric]`
  (one `SeriesMetric` per `(series, model)` pair: `mae`, `rmse`, `n_points`,
  `expected_points`), plus `mean_mae()`/`mean_rmse()` helpers and a
  `to_frame()` that renders the report as a plain pandas `DataFrame`. This
  is a **single evaluation call**, not a training loop — there is no
  concept of "epoch," "step," or a run that improves over iterations to
  track. It's the one closest candidate to something "trackable" among the
  subpackages checked, but its natural comparison unit (one backtest run
  per model/config, compared against another backtest run) is already
  served by comparing two `BacktestReport.to_frame()` outputs directly —
  there's no accumulation-over-time need visible in the actual code today.
- **`automl`**: `compile.py` exists (the `.compile()` → ONNX export path per
  the architecture doc), but there is no training-loop or metrics-reporting
  code in the subpackage today beyond what `compile.py` needs for its ONNX
  export. The issue's "automl's eventual training runs" is explicitly
  forward-looking — "eventual" is the issue's own word — not a description
  of code that exists.
- **`eda`**: `engine.py`/`sketches.py`/`associations.py`/`report.py`
  produce a one-shot **profiling report** (schema, null counts, HLL/KLL
  sketches, association matrix, rendered as a dependency-free HTML/Canvas
  report). This is not a training run in any sense — there's no metric that
  improves over iterations, no hyperparameter to sweep, no "run" concept at
  all. It's a snapshot of a static dataset. Experiment-tracking backends
  (run/step/metric-curve/hyperparameter-versioning tools) don't map onto
  this shape at all; this is arguably the weakest of the three cited
  candidates for needing tracking of any kind.
- **`tune`** (LazyTune, LLM fine-tuning) is the subpackage in the actual ten-
  module list most plausibly shaped like a real training loop (LoRA
  fine-tuning via `peft`/`transformers`, per `pyproject.toml`'s `tune`
  extra), but issue #35 doesn't name it, and as of this evaluation `tune`'s
  training-loop code was not inspected for a metrics-reporting need
  because it's out of scope for this pass; noted here as the most likely
  future real candidate (see part (d)).

**Conclusion for (a):** every subpackage named in the issue surfaces its
results today as a plain dataclass and/or pandas DataFrame, with zero
tracking-backend integration — this part of the issue's premise is
correct. But none of them currently has an actual multi-run,
metric-over-iterations training loop; they produce one-shot reports.

## (b) MLflow vs. W&B: local-only compatibility

Per CLAUDE.md's multi-backend principle, *if* tracking is added, it must
support both as selectable backends, never one hard-coded choice. Both are
Tier-1 permissive per the issue (MLflow Apache-2.0, W&B client MIT) — no
LazyIsolate gating needed for the client libraries.

- **MLflow — local-only: qualifies trivially.** MLflow's default tracking
  backend is a local file store (`mlruns/` on disk, or an explicit
  `mlflow.set_tracking_uri("file:///path")`); the well-known
  `mlflow ui` command reads that same local file store to render a
  dashboard, with no server process, no account, and no network
  dependency required at any point. This has been MLflow's default,
  zero-config behavior since its earliest releases and remains so — a
  bare `import mlflow; mlflow.log_metric(...)` call with no configuration
  writes to a local `./mlruns` directory and never touches the network.
  This is a clean, no-caveats fit for CLAUDE.md's local-only constraint.
- **W&B — local-only: qualifies, with one documented caveat that must be
  called out rather than assumed away.** `wandb.init(mode="offline")` is
  documented by W&B itself (their official support docs, "difference
  wandbinit modes") to write run data to a local file only:
  *"I don't want to depend on the network to send results to your servers
  while executing local operations" ... "wandb.log ... does not block
  network calls."* That's the intended, documented behavior, and it means
  no metric/artifact data leaves the machine unless a separate, explicit
  `wandb sync` command is run later by the user. **However**, research for
  this evaluation surfaced a real, filed W&B GitHub issue
  (`wandb/wandb#2701`, "With WANDB_MODE=offline python client still try to
  sync results") reporting that even with offline mode set, the client
  attempted outbound HTTPS connection retries to W&B's own API endpoint
  (visible in `urllib3` connection-retry warnings) before failing/timing
  out — i.e., no data was exfiltrated (the connection attempts failed), but
  the client did *attempt* outbound network activity in offline mode on at
  least one reported version, which contradicts a strict "fully quiet on
  the wire" expectation for an air-gapped environment. W&B also runs a
  local background service process (`wandb-service`) on `wandb.init()`
  regardless of mode, which listens for local filesystem changes — not
  itself a network concern, but additional local process/resource
  overhead MLflow's plain file-append approach doesn't have.
  **Plain finding: W&B's offline mode does not intentionally phone home —
  by design and by W&B's own documentation, `wandb.log()` never blocks on
  or requires network access — but it is not proven quiet-by-construction
  the way MLflow's local file store is; there is at least one credible,
  filed report of connection *attempts* (not data leaks) still occurring
  in offline mode.** If W&B is ever adopted as a selectable backend here,
  this should be re-verified against the exact pinned `wandb` version at
  implementation time (and ideally tested inside the same sandboxed/
  network-denied environment `dscraft.core.sandbox` already provides for
  `security`/`agent`) rather than trusted on documentation alone.

## (c) Does OpenTelemetry already cover this need? (the central question)

**No — OTel and experiment tracking solve different problems, and a
tracking integration would be complementary, not a competing duplicate
system, PROVIDED it is built to reuse the existing OTel schema rather than
invent a second, parallel metric-naming/schema convention.** This
distinction is worth stating precisely because getting it wrong either way
has a real cost: treating them as redundant would wrongly block a
legitimate future capability; treating them as unrelated and building a
second schema from scratch would violate the architecture doc's own
"shared architecture" decisions (CLAUDE.md: OTel GenAI semantic
conventions are *the* shared schema across security reports, agent
trajectories, and ML leaderboards).

Concretely, reading `dscraft/core/telemetry.py` (311 lines) end to end:

- OTel here is **span/trace-oriented, in-process, and export-optional**.
  `get_tracer()`/`genai_span()`/`set_ml_metric()` create OTel spans and
  attach attributes (`ml.metric.<name>`) to them; without an application-
  configured SDK/exporter, the tracer is a documented no-op — "spans are
  created and attributes/events are accepted, but nothing is exported
  anywhere." Its entire design center is *observability of a running
  process* (a single security probe run, a single agent trajectory, one
  in-flight ML operation), correlated via span parent/child relationships
  within that one execution.
- OTel has **no run/experiment grouping model, no hyperparameter-logging
  concept, no artifact/model-versioning concept, and no cross-run
  comparison UI** — none of which is a gap or oversight in
  `telemetry.py`; it's simply outside OTel's problem domain. `set_ml_metric`
  attaches exactly one metric value to one span at one point in time; there
  is nothing in this module (or in OTel generally) that represents "this
  metric's value across 200 training steps of run A, and how that compares
  to run B and run C, alongside a versioned copy of the model artifact each
  run produced." That is precisely what MLflow/W&B exist to do, and it's
  materially different work: run/experiment identity, a metric-history
  time series *per named run*, hyperparameter key-value logging tied to
  that run, and a UI/API for browsing and comparing many runs.
- Because of that gap, if a real training-loop need shows up later
  (see (d)), the correct design is **not** a second, independent
  metric-naming scheme bolted directly onto MLflow/W&B calls scattered
  through subpackage code — that would indeed be the "second parallel
  telemetry system" this issue rightly worries about. The correct design
  is to keep `dscraft.core.telemetry`'s existing `ml.metric.*` /
  `genai_span` schema as the **single source of truth for what a metric is
  called and what a span represents**, and let an experiment-tracking
  integration be a *consumer* of that same schema — e.g. an OTel
  `SpanProcessor`/exporter that, when installed and configured by the
  calling application, translates `ml.metric.*` span attributes and
  `genai_span` boundaries into `mlflow.log_metric(...)`/`wandb.log(...)`
  calls under the hood. This is the standard, well-established OTel
  pattern (a span processor is exactly OTel's designed extension point for
  "do something else with these spans/attributes") and it means
  `dscraft.core.telemetry`'s existing attribute names and span-naming
  conventions never get a second, competing definition — MLflow/W&B become
  one more *sink* for the same events, not a second event vocabulary.

**Answer to (c), stated plainly: OTel does not already cover the
experiment-tracking need (they solve different problems), but an
integration should be built as a translator/exporter on top of the
existing OTel schema, not as an independent parallel system with its own
metric names and its own instrumentation call sites sprinkled through
subpackage code.**

## (d) Recommendation: defer, with explicit revisit criteria

**Do not build an experiment-tracking integration now**, in `dscraft.core`
or anywhere else. Two independent reasons converge on the same answer:

1. **CLAUDE.md's "two real modules" bar isn't met at zero.** Per part (a),
   none of the subpackages actually named in the issue (`forecast`,
   `automl`, `eda`) has a real, iterative, multi-run training loop today —
   they produce one-shot reports, not metric-over-iterations training
   history. There isn't even *one* concrete, landed consumer with a
   demonstrated tracking need right now, let alone two. Building this now
   would mean designing an integration point against a purely hypothetical
   future shape, which is exactly the premature-abstraction pattern
   CLAUDE.md's shared-infrastructure rule (and this same reasoning already
   applied in the #14 evaluation above) exists to prevent.
2. **The right integration shape depends on a real training loop existing
   first.** Whether the eventual integration point should be
   "wrap `genai_span`/`set_ml_metric` with an optional exporter" (part (c)'s
   recommended shape) or something else entirely can only be validated
   against a real call site — e.g. `tune`'s LoRA fine-tuning loop (per
   `pyproject.toml`'s `tune` extra: `torch`+`transformers`+`peft`), which
   was flagged in part (a) as the most plausible real future candidate
   even though it's outside this issue's own named list and was not
   inspected in this pass. Designing the exporter/translator shape today,
   without that real loop's actual per-step metric names, hyperparameter
   set, and artifact types to validate against, risks building the wrong
   interface.

### Revisit criteria (concrete, so this isn't re-litigated from scratch)

Revisit this decision when **both** of the following are true:

- At least one `dscraft` subpackage has landed real, iterative training-
  loop code that produces a metric history a user would plausibly want to
  compare across multiple runs (most likely `tune`'s fine-tuning loop, or
  a future `automl` hyperparameter-search implementation — not a one-shot
  report like today's `forecast.backtest()` or `eda` profiling).
- That subpackage (or a second one) has an actual, expressed need to
  *compare* runs over time (not just log a single run's final metrics) —
  e.g. comparing LoRA fine-tuning runs across hyperparameter sweeps, or
  AutoML model-selection runs across candidate configs.

At that point, the scoped implementation plan is:

- Add a **new, own extra** (e.g. `dscraft[tracking]`), never a base
  dependency of `core` or any subpackage extra, per the issue's own
  constraint ("must not become a base-install dependency of any
  subpackage").
- Implement it as `dscraft.core.tracking` (or similar), built as an
  **OTel span-processor/exporter layered on top of the existing
  `dscraft.core.telemetry` schema** (per part (c)) — not a parallel
  metric-naming convention — with `mlflow`/`wandb` as selectable backend
  implementations behind one small interface (per CLAUDE.md's multi-
  backend principle: "expose competing frameworks as selectable options,
  never pick-one-over-another").
- Default to MLflow's local file-store backend for any example/test code
  (zero caveats, per part (b)); if W&B's offline mode is exercised in
  tests, re-verify against the exact pinned `wandb` version that offline
  mode doesn't attempt outbound connections (per part (b)'s `wandb#2701`
  finding), ideally inside `dscraft.core.sandbox`'s existing network-denied
  execution mode.

## What was NOT done in this pass

This issue is evaluation-only per its own text and per this task's
instructions. No tracking integration code was written; `dscraft.core`,
`forecast`, `automl`, `eda`, and `pyproject.toml` are unchanged by this
evaluation.
