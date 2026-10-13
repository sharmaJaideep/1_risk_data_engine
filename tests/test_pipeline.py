import pytest

from src.pipeline import PIPELINE, STEPS, execution_order


def test_order_respects_every_dependency():
    position = {step: i for i, step in enumerate(STEPS)}
    for step, deps in PIPELINE.items():
        assert all(position[d] < position[step] for d in deps), step
    assert sorted(STEPS) == sorted(PIPELINE)


def test_expected_shape_of_the_committed_pipeline():
    assert STEPS[0] == "ingest" and STEPS[-1] == "diagnose"
    assert PIPELINE["train"] == ["aggregate"] and PIPELINE["score"] == ["train"]
    assert set(PIPELINE["diagnose"]) == {"stability", "vintage"}  # the agent waits for both branches
    assert PIPELINE["vintage"] == ["validate"]  # vintage does not need the model, so it can run in parallel


def test_order_is_deterministic():
    assert execution_order() == execution_order() == STEPS


def test_cycles_and_undefined_steps_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        execution_order({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="undefined"):
        execution_order({"a": ["ghost"]})


def test_independent_steps_keep_insertion_order():
    assert execution_order({"x": [], "y": [], "z": ["x", "y"]}) == ["x", "y", "z"]


def test_every_pipeline_step_is_runnable_from_the_cli():
    import main

    assert set(STEPS) <= set(main.STEPS)
    import inspect

    source = inspect.getsource(main.run_step)
    for step in STEPS:
        assert f'step == "{step}"' in source, f"main.run_step has no branch for {step!r}"
