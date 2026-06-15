from evox.core.descriptor import PopulationDescriptor
from evox.core.node import Node
from evox.core.population import PopulationDB


def _node(db, fitness, operator, strategy_id, valid=True):
    n = Node(
        id=db.new_id(),
        parent_id=None,
        iteration=0,
        strategy_id=strategy_id,
        operator=operator,
        candidate=[0.0],
        score=fitness,
        fitness=fitness,
        valid=valid,
    )
    return db.add(n)


def test_descriptor_empty_population():
    db = PopulationDB()
    phi = PopulationDescriptor.build(db)
    assert phi.size == 0
    assert phi.num_valid == 0
    assert phi.best_node_id is None
    assert phi.best_fitness == 0.0


def test_descriptor_basic_stats_and_counts():
    db = PopulationDB()
    _node(db, 0.1, "local_refine", "s0")
    _node(db, 0.3, "local_refine", "s0")
    best = _node(db, 0.9, "free_form", "s1")
    _node(db, 0.5, "structural_variation", "s1", valid=False)

    phi = PopulationDescriptor.build(db)
    assert phi.size == 4
    assert phi.num_valid == 3
    assert phi.best_fitness == 0.9
    assert phi.best_node_id == best.id
    assert phi.operator_counts["local_refine"] == 2
    assert phi.strategy_counts["s1"] == 2
    # diversity is the spread of valid fitnesses: 0.9 - 0.1
    assert abs(phi.diversity - 0.8) < 1e-9
    assert phi.mean_fitness > 0


def test_descriptor_is_serialisable():
    db = PopulationDB()
    _node(db, 0.2, "local_refine", "s0")
    d = PopulationDescriptor.build(db).to_dict()
    assert set(["size", "best_fitness", "operator_counts"]).issubset(d.keys())
