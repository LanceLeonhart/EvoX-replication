import pytest

from evox.eval.result import EvalResult
from evox.tasks.base import Task
from evox.tasks.registry import available_tasks, create_task, register_task


def test_builtin_task_registered():
    assert "toy_blackbox" in available_tasks()


def test_create_task_with_params():
    task = create_task("toy_blackbox", {"dim": 3})
    assert task.name == "toy_blackbox"
    assert task.dim == 3
    cand = task.initial_candidate()
    assert len(cand) == 3
    result = task.evaluate(cand)
    assert isinstance(result, EvalResult)
    assert result.valid


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        create_task("does_not_exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        register_task("toy_blackbox", lambda: None)


def test_can_register_and_build_new_task():
    class DummyTask(Task):
        name = "dummy"
        is_maximization = True

        def initial_candidate(self):
            return 0

        def evaluate(self, candidate):
            return EvalResult(score=float(candidate), valid=True)

        def render_task_prompt(self):
            return "dummy"

    register_task("dummy_test_task", DummyTask)
    assert "dummy_test_task" in available_tasks()
    t = create_task("dummy_test_task")
    assert t.evaluate(5).score == 5.0


def test_invalid_candidate_marked_invalid():
    task = create_task("toy_blackbox", {"dim": 4})
    bad = task.evaluate([0.0, 0.0])  # wrong length
    assert not bad.valid
