import json

import pytest

from evox.core.strategy import VALID, Strategy
from evox.llm.parsers import parse_solution_vector, parse_strategy

_STRATEGY_OBJ = {
    "id": "recombine",
    "name": "Recombine",
    "parent_selection": "tournament",
    "operator_weights": {"local_refine": 0.3, "structural_variation": 0.6, "free_form": 0.1},
    "num_inspirations": 2,
    "inspiration_selection": "diverse",
    "exploration": 0.45,
    "notes": "mix promising candidates",
}


def test_parse_strategy_from_clean_json():
    s = parse_strategy(json.dumps(_STRATEGY_OBJ))
    assert isinstance(s, Strategy)
    assert s.id == "recombine"
    assert s.parent_selection == "tournament"
    assert s.num_inspirations == 2
    assert VALID(s)  # parsed object is a legal strategy


def test_parse_strategy_from_fenced_block():
    text = (
        "Sure, here is the strategy:\n"
        "```json\n" + json.dumps(_STRATEGY_OBJ, indent=2) + "\n```\n"
        "Hope that helps!"
    )
    s = parse_strategy(text)
    assert isinstance(s, Strategy)
    assert s.id == "recombine"
    assert s.exploration == 0.45


def test_parse_solution_vector_from_list():
    assert parse_solution_vector("[1.0, 2.5, -3]") == [1.0, 2.5, -3.0]
    # also accepts a wrapped {"candidate": [...]} payload
    assert parse_solution_vector('{"candidate": [0, 1, 2]}') == [0.0, 1.0, 2.0]
    # and a fenced block
    assert parse_solution_vector("```\n[4, 5]\n```") == [4.0, 5.0]


def test_parse_strategy_bad_json_raises():
    with pytest.raises(ValueError):
        parse_strategy("this is not json at all")
    with pytest.raises(ValueError):
        parse_strategy("{ not: valid, json }")


def test_parse_solution_vector_bad_json_raises():
    with pytest.raises(ValueError):
        parse_solution_vector("definitely not a vector")
