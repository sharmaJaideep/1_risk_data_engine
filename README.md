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
  - `data/duckdb/` — Local DuckDB database files and temporary assets.
- `config/` — YAML schema definitions and business rule configurations.
  - `config/schema_config.yaml` — Example schema for `application_train`.
- `src/` — Main Python package.
  - `src/ingestion/` — CSV-to-Parquet pipeline and schema validation helpers.
    - `schema_validator.py` — `pydantic`/YAML-based schema checking.
    - `converter.py` — `polars` CSV -> Parquet converter.
  - `src/metrics/` — PSI/CSI, vintage, and roll-rate computations.
    - `stability.py` — PSI/CSI algorithms.
    - `vintage.py` — Vintage and roll-rate helpers.
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
