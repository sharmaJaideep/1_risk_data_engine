"""The pipeline as a dependency graph.

One definition drives both the local CLI (`python main.py run-all`) and the Airflow DAG in
dags/, so the two cannot drift apart. Each step maps to a `python main.py <step>` command.
"""
from __future__ import annotations

# step -> the steps that must finish first
PIPELINE: dict[str, list[str]] = {
    "ingest": [],
    "drift": ["ingest"],
    "validate": ["drift"],
    "aggregate": ["validate"],
    "train": ["aggregate"],
    "score": ["train"],
    "stability": ["score"],
    "vintage": ["validate"],
    "diagnose": ["stability", "vintage"],
}


def execution_order(graph: dict[str, list[str]] = PIPELINE) -> list[str]:
    """Topological order that is deterministic (ties keep the graph's insertion order)."""
    unknown = {dep for deps in graph.values() for dep in deps} - graph.keys()
    if unknown:
        raise ValueError(f"Steps depend on undefined steps: {sorted(unknown)}")
    order: list[str] = []
    remaining = dict(graph)
    while remaining:
        ready = [step for step, deps in remaining.items() if all(d in order for d in deps)]
        if not ready:
            raise ValueError(f"Dependency cycle among: {sorted(remaining)}")
        order.append(ready[0])
        del remaining[ready[0]]
    return order


STEPS = execution_order()
