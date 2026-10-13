"""Airflow DAG checks. Skipped when Airflow is not installed (it is not a project dependency)."""
import os
from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

from src.pipeline import PIPELINE  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dag():
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.get_dag("risk_data_engine")


def test_dag_graph_matches_the_pipeline_definition(dag):
    edges = {t.task_id: sorted(u.task_id for u in t.upstream_list) for t in dag.tasks}
    assert edges == {step: sorted(deps) for step, deps in PIPELINE.items()}


def test_each_task_runs_its_cli_step(dag):
    for task in dag.tasks:
        assert f"main.py {task.task_id}" in task.bash_command


def test_scheduling_safety_settings(dag):
    assert dag.catchup is False and dag.max_active_runs == 1  # runs share files; never overlap
    assert dag.default_args["retries"] == 2


def test_diagnose_alerts_on_critical_but_never_retries(dag):
    task = dag.get_task("diagnose")
    assert "--fail-on critical" in task.bash_command
    assert task.retries == 0  # the verdict is deterministic, so a retry only delays the alert
    assert all(dag.get_task(s).retries == 2 for s in PIPELINE if s != "diagnose")
