"""Airflow DAG for the Risk Data Engine.

Runs the same dependency graph as `python main.py run-all` (src/pipeline.py), one Airflow task
per step, so the local CLI and the scheduled pipeline cannot drift apart:

    ingest -> drift -> validate -> aggregate -> train -> score -> stability -+-> diagnose
                          \\-> vintage ------------------------------------+

Failures retry twice, except `diagnose`: it runs with `--fail-on critical`, so a CRITICAL
portfolio verdict fails the task and triggers whatever alerting the Airflow deployment has
configured. Its verdict is deterministic, so retrying would only delay the alert.

Environment:
  RISK_ENGINE_HOME    project root (default: the repo this file lives in)
  RISK_ENGINE_PYTHON  interpreter with the project's dependencies (default: `python`)
  ANTHROPIC_API_KEY   optional; enables the LLM narrative in `diagnose`
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

ROOT = Path(os.environ.get("RISK_ENGINE_HOME", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from src.pipeline import PIPELINE  # noqa: E402

PYTHON = os.environ.get("RISK_ENGINE_PYTHON", "python")
STEP_FLAGS = {"diagnose": "--fail-on critical"}
NO_RETRY = {"diagnose"}  # a bad verdict is a result, not a transient failure

with DAG(
    dag_id="risk_data_engine",
    description="Ingest, validate, model and monitor the credit portfolio",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,  # steps share files in data/processed; never overlap two runs
    default_args={
        "owner": "risk-analytics",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
    },
    tags=["risk", "monitoring"],
    doc_md=__doc__,
) as dag:
    tasks = {
        step: BashOperator(
            task_id=step,
            bash_command=f"cd {ROOT} && {PYTHON} main.py {step} {STEP_FLAGS.get(step, '')}".strip(),
            **({"retries": 0} if step in NO_RETRY else {}),
        )
        for step in PIPELINE
    }
    for step, upstream in PIPELINE.items():
        for dep in upstream:
            tasks[dep] >> tasks[step]
