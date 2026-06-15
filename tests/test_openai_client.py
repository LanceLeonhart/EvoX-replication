"""Tests for the OpenAI backend using fake response objects — never the real API."""

import json

import pytest

from evox.core.engine import Engine
from evox.core.strategy import Strategy
from evox.llm.client import OpenAIClient, SolutionRequest
from evox.logging.event_log import EventLog
from evox.tasks.registry import create_task


# ── fake Responses API ─────────────────────────────────────────────────────
class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, output_text, usage):
        self.output_text = output_text
        self.usage = usage


class _FakeResponses:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._replies.pop(0) if self._replies else self._replies_default()
        return _FakeResponse(text, _FakeUsage(11, 7))

    @staticmethod
    def _replies_default():
        return "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"


class _FakeOpenAI:
    def __init__(self, replies):
        self.responses = _FakeResponses(replies)


def _vector_request(dim=6):
    task = create_task("toy_blackbox", {"dim": dim})
    return SolutionRequest(
        operator="local_refine",
        parent_candidate=task.initial_candidate(),
        inspirations=[],
        task_prompt=task.render_task_prompt(),
        candidate_schema=task.candidate_schema(),
        exploration=0.15,
        strategy_id="s0",
        seed=1,
    )


def _strategy():
    return Strategy(
        id="exploit_local",
        name="Exploit",
        parent_selection="best",
        operator_weights={"local_refine": 0.85, "structural_variation": 0.12, "free_form": 0.03},
        num_inspirations=1,
        inspiration_selection="best",
        exploration=0.15,
    )


# ── tests ──────────────────────────────────────────────────────────────────
def test_openai_client_generates_and_parses_vector():
    fake = _FakeOpenAI(["[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]"])
    client = OpenAIClient(client=fake, model="gpt-5-mini")
    resp = client.generate_solution(_vector_request(), "PROMPT")

    assert resp.candidate == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert resp.parse_error is None
    assert resp.model == "gpt-5-mini"
    assert resp.prompt_tokens == 11 and resp.completion_tokens == 7
    # low-cost defaults were passed to the Responses API
    kw = fake.responses.calls[0]
    assert kw["model"] == "gpt-5-mini"
    assert kw["reasoning"] == {"effort": "minimal"}
    assert kw["text"] == {"verbosity": "low"}
    assert kw["max_output_tokens"] == 512
    assert kw["input"] == "PROMPT"


def test_openai_client_parse_failure_marked_invalid_not_crash():
    fake = _FakeOpenAI(["sorry, I cannot comply"])
    client = OpenAIClient(client=fake)
    resp = client.generate_solution(_vector_request(), "PROMPT")

    assert resp.candidate is None
    assert resp.parse_error  # reason captured
    assert resp.raw_text == "sorry, I cannot comply"  # raw output preserved for debugging


def test_openai_missing_key_raises_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        OpenAIClient()  # no injected client, no key
    assert "OPENAI_API_KEY" in str(exc.value)


def test_runner_openai_mode_missing_key_raises_no_silent_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from evox.eval.runner import run_from_config

    config = {
        "name": "openai_test",
        "task": {"name": "toy_blackbox", "params": {"dim": 4}},
        "engine": {"budget_T": 4, "window_W": 2, "tau": 0.05, "seed": 1},
        "strategy": _strategy().to_dict(),
        "llm": {"mode": "openai", "model": "gpt-5-mini"},
        "output": {"dir": str(tmp_path)},
    }
    with pytest.raises(RuntimeError) as exc:
        run_from_config(config, run_dir=str(tmp_path))
    assert "OPENAI_API_KEY" in str(exc.value)


def test_engine_runs_end_to_end_with_fake_openai():
    # A run with valid generations completes and logs generation metadata.
    fake = _FakeOpenAI(["[0.1, 0.1, 0.1, 0.1, 0.1]"] * 50)
    client = OpenAIClient(client=fake, model="gpt-5-mini")
    log = EventLog()
    eng = Engine(
        task=create_task("toy_blackbox", {"dim": 5}),
        initial_strategy=_strategy(),
        client=client,
        budget_T=6,
        window_W=3,
        tau=-1.0,
        seed=7,
        event_log=log,
    )
    summary = eng.run()
    assert summary.population_size == 7  # seed + 6

    gen_events = [e for e in log.of_type("iteration") if "generation" in e]
    assert gen_events, "iteration events should carry generation metadata"
    meta = gen_events[-1]["generation"]
    assert meta["model"] == "gpt-5-mini"
    assert meta["raw_output"]  # raw model output logged
    assert "prompt_preview" in meta


def test_engine_continues_through_parse_failures_with_fake_openai():
    fake = _FakeOpenAI(["not a vector"] * 50)
    client = OpenAIClient(client=fake, model="gpt-5-mini")
    log = EventLog()
    eng = Engine(
        task=create_task("toy_blackbox", {"dim": 5}),
        initial_strategy=_strategy(),
        client=client,
        budget_T=4,
        window_W=2,
        tau=-1.0,
        seed=1,
        event_log=log,
    )
    summary = eng.run()
    # run did not crash; the generated nodes are invalid, only the seed is valid
    assert summary.population_size == 5
    invalid_gen = [
        e for e in log.of_type("iteration")
        if e.get("generation") and e["generation"]["parse_error"]
    ]
    assert len(invalid_gen) == 4
    assert all(e["valid"] is False for e in invalid_gen)
