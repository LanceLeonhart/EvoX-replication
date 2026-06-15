"""Tests for real outer-loop strategy generation (V2) using fake OpenAI
responses. The real API is never called."""

import json

import pytest

from evox.core.engine import Engine
from evox.core.strategy import VALID, Strategy
from evox.llm.client import (
    CompositeLLMClient,
    MockLLMClient,
    OpenAIClient,
    StrategyRequest,
)
from evox.logging.event_log import EventLog
from evox.tasks.registry import create_task


# ── strategy payloads ──────────────────────────────────────────────────────
_VALID_STRATEGY = {
    "id": "recombine",
    "name": "Recombine",
    "parent_selection": "tournament",
    "operator_weights": {"local_refine": 0.3, "structural_variation": 0.6, "free_form": 0.1},
    "num_inspirations": 2,
    "inspiration_selection": "diverse",
    "exploration": 0.45,
    "notes": "mix promising candidates",
}
_INVALID_SCHEMA_STRATEGY = dict(_VALID_STRATEGY, parent_selection="psychic")

_VALID_JSON = json.dumps(_VALID_STRATEGY)
_FENCED_JSON = "Here you go:\n```json\n" + json.dumps(_VALID_STRATEGY, indent=2) + "\n```\nThanks!"
_INVALID_JSON = "sorry, I can't help with that"
_INVALID_SCHEMA_JSON = json.dumps(_INVALID_SCHEMA_STRATEGY)
_VECTOR_REPLY = "[0.1, 0.1, 0.1, 0.1, 0.1]"


# ── fakes ───────────────────────────────────────────────────────────────────
class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Resp:
    def __init__(self, text):
        self.output_text = text
        self.usage = _Usage(13, 9)


class _ReplyResponses:
    """Always returns the next reply (list pops; str repeats) regardless of input."""

    def __init__(self, replies):
        self._replies = list(replies) if isinstance(replies, list) else None
        self._fixed = None if isinstance(replies, list) else replies
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._fixed if self._fixed is not None else (
            self._replies.pop(0) if self._replies else "[]"
        )
        return _Resp(text)


class _ReplyOpenAI:
    def __init__(self, replies):
        self.responses = _ReplyResponses(replies)


class _RoutingResponses:
    """Routes by prompt content: strategy prompts vs solution prompts."""

    def __init__(self, solution_reply, strategy_reply):
        self.solution_reply = solution_reply
        self.strategy_reply = strategy_reply
        self.solution_calls = []
        self.strategy_calls = []

    def create(self, **kwargs):
        inp = kwargs.get("input", "")
        if "Legal strategy vocabulary" in inp or "Strategy schema" in inp:
            self.strategy_calls.append(kwargs)
            text = self.strategy_reply
        else:
            self.solution_calls.append(kwargs)
            text = self.solution_reply
        return _Resp(text)


class _RoutingOpenAI:
    def __init__(self, solution_reply=_VECTOR_REPLY, strategy_reply=_VALID_JSON):
        self.responses = _RoutingResponses(solution_reply, strategy_reply)


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


def _request():
    return StrategyRequest(
        current_strategy=_strategy(), descriptor={}, tried_signatures=[], seed=3
    )


def _engine(client, tau, T=12, W=3, seed=7, dim=5):
    return Engine(
        task=create_task("toy_blackbox", {"dim": dim}),
        initial_strategy=_strategy(),
        client=client,
        budget_T=T,
        window_W=W,
        tau=tau,
        seed=seed,
        event_log=EventLog(),
    )


# ── direct propose_strategy tests ───────────────────────────────────────────
def test_clean_json_strategy_parsed_validated_accepted():
    client = OpenAIClient(client=_ReplyOpenAI(_VALID_JSON), model="gpt-5-mini")
    resp = client.propose_strategy(_request(), "PROMPT")
    assert resp.strategy.id == "recombine"
    assert VALID(resp.strategy)
    assert resp.used_fallback is False
    assert resp.attempts == 1
    assert resp.model == "gpt-5-mini"
    assert resp.prompt_tokens == 13 and resp.completion_tokens == 9


def test_fenced_chatty_json_strategy_accepted():
    client = OpenAIClient(client=_ReplyOpenAI(_FENCED_JSON))
    resp = client.propose_strategy(_request(), "PROMPT")
    assert resp.strategy.id == "recombine"
    assert resp.used_fallback is False


def test_invalid_json_retries_then_falls_back():
    fake = _ReplyOpenAI(_INVALID_JSON)  # every attempt is unparseable
    client = OpenAIClient(client=fake, strategy_retries=1)  # -> 2 attempts
    resp = client.propose_strategy(_request(), "PROMPT")
    assert resp.used_fallback is True
    assert resp.attempts == 2
    assert len(fake.responses.calls) == 2  # both attempts hit the API
    assert len(resp.errors) == 2 and all("parse" in e for e in resp.errors)
    assert VALID(resp.strategy)  # fallback strategy is always valid


def test_invalid_schema_retries_then_falls_back():
    fake = _ReplyOpenAI(_INVALID_SCHEMA_JSON)
    client = OpenAIClient(client=fake, strategy_retries=2)  # -> 3 attempts
    resp = client.propose_strategy(_request(), "PROMPT")
    assert resp.used_fallback is True
    assert resp.attempts == 3
    assert all("invalid" in e for e in resp.errors)
    assert VALID(resp.strategy)


def test_retry_count_is_configurable():
    fake = _ReplyOpenAI(_INVALID_JSON)
    client = OpenAIClient(client=fake, strategy_retries=0)  # -> 1 attempt only
    resp = client.propose_strategy(_request(), "PROMPT")
    assert resp.attempts == 1
    assert len(fake.responses.calls) == 1
    assert resp.used_fallback is True


# ── engine-level tests (mode routing) ───────────────────────────────────────
def test_mock_strategy_mode_never_calls_responses_api():
    # openai solution + mock strategy: API used for solutions, never for strategy
    fake = _RoutingOpenAI()
    sol = OpenAIClient(client=fake, model="gpt-5-mini")
    composite = CompositeLLMClient(solution_client=sol, strategy_client=MockLLMClient())
    eng = _engine(composite, tau=10.0)  # high tau -> stagnation every window
    summary = eng.run()
    assert summary.num_strategy_switches >= 1          # mock proposer still switches
    assert fake.responses.strategy_calls == []         # but no strategy API call
    assert fake.responses.solution_calls               # solutions did call the API


def test_openai_strategy_mode_calls_api_on_stagnation():
    # mock solution + openai strategy: API used only for strategy, on stagnation
    fake = _RoutingOpenAI(strategy_reply=_VALID_JSON)
    strat = OpenAIClient(client=fake, model="gpt-5-mini")
    composite = CompositeLLMClient(solution_client=MockLLMClient(), strategy_client=strat)
    eng = _engine(composite, tau=10.0)
    summary = eng.run()
    assert fake.responses.solution_calls == []         # solutions were mock
    assert fake.responses.strategy_calls               # strategy hit the API
    assert summary.num_strategy_switches >= 1
    # accepted strategy came from the model
    switches = eng.log.of_type("strategy_switch")
    assert any(s["to_strategy"] == "recombine" for s in switches)
    assert all(s["strategy_generation"]["used_fallback"] is False for s in switches)


def test_strategy_api_not_called_without_stagnation():
    fake = _RoutingOpenAI()
    strat = OpenAIClient(client=fake, model="gpt-5-mini")
    composite = CompositeLLMClient(solution_client=MockLLMClient(), strategy_client=strat)
    eng = _engine(composite, tau=-1.0)  # never stagnant -> no strategy generation
    eng.run()
    assert fake.responses.strategy_calls == []


def test_population_preserved_across_real_strategy_switch():
    fake = _RoutingOpenAI(strategy_reply=_VALID_JSON)
    strat = OpenAIClient(client=fake, model="gpt-5-mini")
    composite = CompositeLLMClient(solution_client=MockLLMClient(), strategy_client=strat)
    eng = _engine(composite, tau=10.0)
    eng.run()
    sizes = [e["population_size"] for e in eng.log.events() if "population_size" in e]
    assert sizes == sorted(sizes)
    for sw in eng.log.of_type("strategy_switch"):
        assert sw["population_size_after"] == sw["population_size_before"]


def test_engine_survives_strategy_fallback_via_openai():
    # strategy model always emits garbage -> fallback used, run completes
    fake = _RoutingOpenAI(strategy_reply=_INVALID_JSON)
    strat = OpenAIClient(client=fake, model="gpt-5-mini", strategy_retries=1)
    composite = CompositeLLMClient(solution_client=MockLLMClient(), strategy_client=strat)
    eng = _engine(composite, tau=10.0)
    summary = eng.run()
    assert summary.population_size == 12 + 1
    switches = eng.log.of_type("strategy_switch")
    assert switches and all(s["strategy_generation"]["used_fallback"] for s in switches)


# ── runner / config tests ───────────────────────────────────────────────────
def test_runner_strategy_openai_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from evox.eval.runner import run_from_config

    config = {
        "name": "openai_strategy_test",
        "task": {"name": "toy_blackbox", "params": {"dim": 4}},
        "engine": {"budget_T": 4, "window_W": 2, "tau": 0.05, "seed": 1},
        "strategy": _strategy().to_dict(),
        "llm": {"solution_mode": "mock", "strategy_mode": "openai", "model": "gpt-5-mini"},
        "output": {"dir": str(tmp_path)},
    }
    with pytest.raises(RuntimeError) as exc:
        run_from_config(config, run_dir=str(tmp_path))
    assert "OPENAI_API_KEY" in str(exc.value)


def test_legacy_mode_keeps_strategy_mock(monkeypatch):
    # legacy `mode: openai` must NOT enable real strategy generation
    from evox.eval.runner import _make_client
    from evox.logging.usage import UsageTracker

    fake = _RoutingOpenAI()

    # patch OpenAIClient construction in the runner to inject our fake transport
    import evox.eval.runner as runner_mod
    real_cls = runner_mod.OpenAIClient
    monkeypatch.setattr(
        runner_mod,
        "OpenAIClient",
        lambda **kw: real_cls(client=fake, **{k: v for k, v in kw.items() if k != "api_key"}),
    )

    client = _make_client({"llm": {"mode": "openai"}}, UsageTracker())
    # strategy backend should be the mock, not the openai client
    from evox.llm.client import CompositeLLMClient, MockLLMClient
    assert isinstance(client, CompositeLLMClient)
    assert isinstance(client.strategy_client, MockLLMClient)
