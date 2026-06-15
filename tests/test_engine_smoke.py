import math

from evox.core.engine import Engine
from evox.core.progress import is_stagnant, strategy_score, window_delta
from evox.core.strategy import Strategy
from evox.llm.client import MockLLMClient
from evox.logging.event_log import EventLog
from evox.tasks.registry import create_task


def _initial_strategy():
    return Strategy(
        id="exploit_local",
        name="Exploit",
        parent_selection="best",
        operator_weights={"local_refine": 0.85, "structural_variation": 0.12, "free_form": 0.03},
        num_inspirations=1,
        inspiration_selection="best",
        exploration=0.15,
    )


def _engine(tau, T=12, W=3, seed=7):
    task = create_task("toy_blackbox", {"dim": 5})
    log = EventLog()
    return Engine(
        task=task,
        initial_strategy=_initial_strategy(),
        client=MockLLMClient(),
        budget_T=T,
        window_W=W,
        tau=tau,
        seed=seed,
        event_log=log,
    )


def test_engine_runs_and_population_size_is_correct():
    eng = _engine(tau=-1.0, T=12, W=3)  # no stagnation -> no switches
    summary = eng.run()
    # seed node + T inner nodes
    assert summary.population_size == 12 + 1
    assert summary.iterations == 12
    assert summary.num_strategy_switches == 0
    assert summary.best_fitness > 0


def test_window_summaries_logged():
    eng = _engine(tau=-1.0, T=12, W=3)
    eng.run()
    windows = eng.log.of_type("window_summary")
    assert len(windows) == 12 // 3
    for w in windows:
        assert "delta" in w and "J" in w and "descriptor" in w


def test_stagnation_triggers_at_least_one_switch():
    # high tau => every window counts as stagnant => switch attempted
    eng = _engine(tau=10.0, T=12, W=3)
    summary = eng.run()
    assert summary.num_strategy_switches >= 1
    switches = eng.log.of_type("strategy_switch")
    assert len(switches) >= 1
    # more than one distinct strategy was active over the run
    assert len(set(summary.strategy_ids)) >= 2


def test_population_never_reset_across_switches():
    eng = _engine(tau=10.0, T=12, W=3)
    eng.run()
    # population size is monotonically non-decreasing across every logged event
    sizes = [e["population_size"] for e in eng.log.events() if "population_size" in e]
    assert sizes == sorted(sizes)
    # and switch events show population is preserved, not reset
    for sw in eng.log.of_type("strategy_switch"):
        assert sw["population_size_after"] == sw["population_size_before"]
        assert sw["population_size_after"] > 0


def test_progress_formulas():
    assert window_delta(0.2, 0.5) == 0.3
    expected = 0.3 * math.log(1 + 0.2) / math.sqrt(4)
    assert abs(strategy_score(0.3, 0.2, 4) - expected) < 1e-12
    assert is_stagnant(0.01, 0.05)
    assert not is_stagnant(0.10, 0.05)
