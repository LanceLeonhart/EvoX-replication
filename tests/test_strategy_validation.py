from evox.core.strategy import Strategy, VALID, validate_strategy


def make(**overrides):
    base = dict(
        id="s",
        name="S",
        parent_selection="best",
        operator_weights={"local_refine": 1.0},
        num_inspirations=1,
        inspiration_selection="best",
        exploration=0.2,
    )
    base.update(overrides)
    return Strategy(**base)


def test_valid_strategy_passes():
    ok, reasons = validate_strategy(make())
    assert ok and reasons == []
    assert VALID(make())


def test_bad_parent_selection_rejected():
    ok, reasons = validate_strategy(make(parent_selection="psychic"))
    assert not ok
    assert any("parent_selection" in r for r in reasons)


def test_unknown_operator_rejected():
    ok, reasons = validate_strategy(make(operator_weights={"telepathy": 1.0}))
    assert not ok
    assert any("unknown operator" in r for r in reasons)


def test_zero_weight_sum_rejected():
    ok, reasons = validate_strategy(make(operator_weights={"local_refine": 0.0}))
    assert not ok
    assert any("sum to 0" in r for r in reasons)


def test_exploration_out_of_range_rejected():
    assert not VALID(make(exploration=1.5))
    assert not VALID(make(exploration=-0.1))


def test_negative_inspirations_rejected():
    assert not VALID(make(num_inspirations=-1))


def test_signature_distinguishes_behaviour():
    a = make()
    b = make(id="other", name="Other")  # same behaviour, different identity
    assert a.signature() == b.signature()
    c = make(exploration=0.9)
    assert a.signature() != c.signature()
