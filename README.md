# High-Scale Risk Data Engine & Migration Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](#)

## System Overview

This repository implements a High-Scale Risk Data Engine & Migration Framework designed for credit portfolio analytics on multi-table datasets (e.g., the Kaggle Home Credit Default Risk dataset). The engine combines Polars (fast, memory-efficient DataFrames with lazy evaluation) and DuckDB (zero-copy SQL analytics) to deliver high-throughput ingestion, schema reconciliation, and stability metric generation (PSI/CSI), wrapped with Agentic AI diagnostics for autonomous monitoring and root-cause reporting.

Core capabilities:

- Schema reconciliation and validation across monthly cohorts using `pydantic`-driven schemas.
- Fast CSV -> compressed Parquet ingestion using `polars` and `pyarrow`.
- DuckDB zero-copy access to Parquet files for SQL analytics without full data copies.
- Automated stability metric generation: Population Stability Index (PSI), Characteristic Stability Index (CSI), Roll-Rates, and Vintage analysis.
- Agentic AI diagnostic layer that inspects metric breaches, produces human-readable findings, and suggests remediation.

## System Architecture

ASCII flow diagram (data flow):

```
Kaggle Raw CSVs
     |
     v
Compressed Parquet (data/processed)
     |
     v
Hierarchical Rollup (Level 2 -> Level 1 -> Level 0)
     |
     v
DuckDB / Polars Engine (zero-copy & lazy execution)
     |
     v
Metric Engine (PSI / CSI / Roll Rate / Vintage)
     |
     v
Agentic AI Diagnostic Layer (autonomous briefs + alerts)
```

## Key Technical Features

- Performance Processing: Polars `LazyFrame`s for query planning and parallel execution, combined with DuckDB zero-copy reads on Parquet for fast sub-second aggregations on moderate cluster/desktop hardware.
- Schema Reconciliation: Versioned YAML schemas and `pydantic` models drive deterministic column validation, automated drift detection, and transformation hints when cohorts diverge.
- Portfolio Stability Analytics: End-to-end automated computation of PSI/CSI with configurable binning, null handling, and threshold-driven alerts for production monitoring.
- Agentic AI Monitoring: Diagnostic agents that use LLMs (via `pydantic-ai` or other orchestration) to examine metric trends, surface likely root causes, and produce markdown summaries for downstream teams.

## Directory Blueprint

Project layout (high level):

- `data/` — Holds raw and processed data (gitignored). 
  - `data/raw/` — Raw CSVs from Kaggle (place originals here).
  - `data/processed/` — Compressed Parquet outputs from ingestion.
  - `data/duckdb/` — Optional persistent DuckDB file, created on demand by `DuckDBEngine()` (the pipeline itself runs in memory).
- `config/` — YAML schema definitions and business rule configurations.
  - `config/stability.yaml` — Cohort definitions for PSI/CSI: train vs test, and older vs more recent records per table.
  - `config/thresholds.yaml` — Roll-rate alert thresholds for the diagnostic agent, with per-source overrides (bureau, credit_card, pos_cash).
  - `config/schemas/<table>/v<N>.yaml` — Versioned column schemas (types, nullability, primary key) for each of the 8 Home Credit tables. Versions are immutable and the highest N is current. Add one with `python scripts/generate_schema.py --bump <table>` and review it with `python main.py schema-diff --table <table>`.
- `src/` — Main Python package.
  - `src/ingestion/` — CSV-to-Parquet pipeline and schema validation helpers.
    - `schema_validator.py` — `pydantic`/YAML-based schema checking.
    - `converter.py` — `polars` CSV -> Parquet converter.
  - `src/modeling/` — `pd_model.py`: PD model training, evaluation and scoring.
  - `src/pipeline.py` — The step dependency graph shared by the CLI and the Airflow DAG.
  - `src/metrics/` — PSI/CSI, vintage, and roll-rate computations.
    - `stability.py` — PSI (score) and CSI (per-bin characteristic) drift detection, driven by `config/stability.yaml`.
    - `vintage.py` — Roll-rate (by account count, and by balance for credit card) and vintage analysis for bureau, credit card and POS cash accounts.
  - `src/db/` — DuckDB connector and zero-copy helpers.
    - `duckdb_engine.py` — Simple DuckDB connection wrapper.
  - `src/agents/` — Agentic diagnostic components.
    - `diagnostic_agent.py` — Lightweight diagnostic agent interface.
  - `src/utils/` — Logging and small utilities.
    - `logger.py` — Project logger factory.
- `tests/` — Pytest suite for core units.
- `notebooks/` — Exploratory notebooks for analysis and EDA.
- `main.py` — CLI entry point for running ingestion / diagnostics.
- `requirements.txt`, `pyproject.toml`, `.gitignore`

## Getting Started & Quickstart

Prerequisites: Python 3.10+ recommended, and an environment with sufficient memory and disk for multi-GB datasets.

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Place CSVs

Download or copy the Kaggle Home Credit CSVs into `data/raw/` (e.g., `application_train.csv`, `bureau.csv`, `installments_payments.csv`, ...).

4. Run ingestion + metrics

```bash
# Convert CSVs to Parquet and run simple diagnostics
python main.py
```

Note: `main.py` is a lightweight CLI bootstrap — extend it to orchestrate full ingestion, DuckDB registration, and automated diagnostics for production use.

## Mathematical Formulation

Population Stability Index (PSI) is computed across $n$ bins as:

$$
PSI = \sum_{i=1}^{n} (A_i - E_i) \ln\left(\frac{A_i}{E_i}\right)
$$

where $A_i$ is the proportion of the actual (new) population in bin $i$, and $E_i$ is the proportion of the expected (baseline) population in bin $i$.

Common interpretation thresholds:

- PSI < 0.1: negligible change
- 0.1 ≤ PSI < 0.25: moderate change (investigate)
- PSI ≥ 0.25: major shift (likely model recalibration required)

## Target Capabilities & Metrics

Expected performance benefits vs. legacy Pandas workflows (representative):

- CSV -> Parquet conversion: typically 5–15x faster using `polars` I/O and multicore writes.
- Grouped aggregations and rollups: often 10–50x faster using `polars` lazy execution and DuckDB vectorized SQL for zero-copy scans.
- Memory footprint: Polars' Apache Arrow-based columnar memory layout reduces peak memory compared to row-oriented Pandas in many aggregation workloads.

Actual gains depend on dataset size, I/O bandwidth, CPU cores, and the nature of operations. Use real benchmarks in your environment for precise planning.

## Operational Considerations

- Data governance: Keep raw CSVs in `data/raw` and store production Parquet/duckdb artifacts on durable storage (S3, GCS) for reproducibility.
- Monitoring: Hook PSI/CSI threshold alerts into your observability stack (Prometheus, PagerDuty, Slack) via the Agentic AI layer.
- Security: Keep secrets out of repo; use `.env` + `python-dotenv` for local testing and a secrets manager in production.

## Extending the Framework

- Add batch or workflow orchestration (Airflow, Prefect, Dagster) to schedule ingestion, materialization, and diagnostics.
- Expand `src/agents/` to integrate an LLM provider and richer root-cause analysis pipelines.
- Add data quality checks (null-rate, cardinality, uniqueness) and automatic transformation recipes when schema drift is detected.

---

If you want, I can now:

- Implement a full ingestion runner that converts all `data/raw/*.csv` to `data/processed/*.parquet` and registers them in a `DuckDB` instance.
- Add CI checks and a benchmark notebook that measures Polars vs. Pandas for a few key aggregation queries.

Choose one and I will proceed.

## Predictive model

`python main.py train` fits a gradient-boosted probability-of-default model on the master record (`src/modeling/pd_model.py`); `python main.py score` scores the `application_test` population.
On a stratified 20% hold-out it reaches **AUC 0.77, Gini 0.54, KS 0.41**, and predicted default rates match observed rates across score deciles.
The hold-out is random, not out-of-time (Home Credit has no dates), so treat these as an optimistic bound. Metrics are written to `data/models/model_metrics.json`.

The model closes the monitoring loop: PSI on the score itself (development hold-out vs the scored population), and a model-health check in the diagnostic agent (discrimination floors and calibration, set in `config/thresholds.yaml`).

## Orchestration and deployment

`src/pipeline.py` defines the steps and their dependencies once. `python main.py run-all` follows it, and so does the Airflow DAG in `dags/risk_engine_dag.py` (one task per step, retries, `max_active_runs=1`).
The DAG's `diagnose` task runs with `--fail-on critical`, so a CRITICAL verdict fails the task and fires whatever alerting your Airflow has; it is the one task that does not retry, because its verdict is deterministic.
The DAG was run end to end under Airflow 2.10 (all steps succeeded; `diagnose` failed on the CRITICAL verdict, as designed). Airflow is not a project dependency: install it separately to use the DAG, and set `RISK_ENGINE_PYTHON` to the interpreter that has this project's requirements.

**Container.** The `Dockerfile` builds one image for every step: `docker run -v $PWD/data:/data <image> run-all`. It was written but **not built or tested** (no Docker in the development environment).

**Cloud.** Nothing here is deployed. A typical AWS mapping: image in ECR; data in S3 (the pipeline reads local paths today, so mount the bucket or sync it to disk, or extend the readers to `s3://` URIs via DuckDB's `httpfs`); the DAG on MWAA, with each step as an ECS/Fargate task; `ANTHROPIC_API_KEY` from Secrets Manager; Airflow task failures to your alerting channel.

**Spark.** Not used, deliberately. The whole pipeline (41M-row tables) runs in about 35 seconds on a laptop, and DuckDB/Polars beat Spark at this size because there is no cluster start-up or shuffle cost. Spark earns its place when data no longer fits one machine (hundreds of GB and up). The aggregation SQL in `src/metrics/aggregators.py` is close to portable to Spark SQL.

## Testing

```
python -m pytest --cov          # 133 tests, ~4 s; fails below 90% line coverage (currently 99%)
```

Tests build their own tiny datasets; the few that check the real `data/processed` files skip when it is absent.
CI (`.github/workflows/ci.yml`) runs the same command on Python 3.10 and 3.11.
Beyond line coverage, key logic (PSI maths, roll-rate and vintage definitions, aggregators, agent escalation, schema/versioning, converter, CLI) is covered by behavioural assertions, and 17 of 18 deliberately planted bugs were caught by the suite; the 18th was an equivalent change.

## Stability monitoring (PSI / CSI)

`python main.py stability` compares a baseline cohort with a target cohort for every feature, using baseline-derived bins.
Numeric features use quantile bins with open-ended edges, categorical features use one bin per category plus `__OTHER__`, and missing values get their own `__NULL__` bin.
The report gives each feature's PSI, a status (stable < 0.10 <= moderate < 0.25 <= significant) and the bin that drives the shift (`top_driver`).

Cohorts are set in `config/stability.yaml`:
- `application_train_vs_test` alerts: is the population being scored like the one developed on? (EXT_SOURCE_* are tagged as scores.)
- Time-split analyses (older 70% vs more recent 30% of each child table) are reported but informational.

**Limitation:** Home Credit has no calendar dates. Every time column is relative to each customer's own application date, so these splits show older-vs-recent *history*, not calendar drift, and they differ structurally (product mix, fields collected). With real dates, set `alert: true` for them.

## Benchmark: pandas vs the DuckDB pipeline

Master Analytical Record build (5 input tables, ~41M rows in the two largest, 15 aggregates, left-joined onto 307k applications).
Reproduce with `python scripts/benchmark_pipeline.py --runs 3`. Each run is a fresh process; the pandas output is checked against the pipeline output.

| | pandas | DuckDB pipeline |
|---|---|---|
| Wall time (median of 3) | 6.5 s | 1.3 s |
| Peak memory, RSS (median of 3) | 3.3 GB | 2.1 GB |

About 5x faster and 38% lower peak memory on a 10-core, 16 GB Apple-silicon laptop. The pandas baseline reads only the columns it needs, so it is a fair, not a naive, comparison. These are single-machine numbers on this dataset; they will differ elsewhere.
