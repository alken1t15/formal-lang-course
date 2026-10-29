"""Tests for tensor-based context-free reachability."""

from copy import deepcopy
from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from pyformlang.cfg import CFG
from scipy.sparse import isspmatrix_csr

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq
from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq
from test_hellings_based_cfpq import reference_cfpq


def test_tensor_balanced_paths_and_internal_vertices():
    graph = DiGraph()
    graph.add_edges_from(
        (node, node + 1, {"label": label})
        for node, label in enumerate(["a", "a", "b", "b"])
    )
    rsm = ebnf_to_rsm("S -> a S b | $")
    assert tensor_based_cfpq(rsm, graph, {0, 1}, {3, 4}) == {(0, 4), (1, 3)}
    assert tensor_based_cfpq(rsm, graph, {0}, {3}) == set()


def test_tensor_nullable_box_calls_and_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_edge(-7, 90, label="b")
    graph.add_node(999)
    rsm = ebnf_to_rsm("S -> A B\nA -> a | $\nB -> b | $")
    assert tensor_based_cfpq(rsm, graph) == {(-7, -7), (90, 90), (999, 999), (-7, 90)}
    assert tensor_based_cfpq(rsm, graph, {-7}, {90}) == {(-7, 90)}


def test_tensor_keeps_states_of_different_boxes_separate():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    rsm = ebnf_to_rsm("S -> A B\nA -> a\nB -> b")
    assert tensor_based_cfpq(rsm, graph) == {(10, 30)}
    assert tensor_based_cfpq(rsm, graph, {10}, {20}) == set()


def test_tensor_raw_box_named_edge_is_not_a_completed_call():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="A")
    graph.add_edge(20, 30, label="b")
    rsm = ebnf_to_rsm("S -> A b\nA -> a")
    assert tensor_based_cfpq(rsm, graph) == set()
    graph.add_edge(10, 20, label="a")
    assert tensor_based_cfpq(rsm, graph) == {(10, 30)}


def test_tensor_recursive_cycle_reaches_a_fixed_point():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")
    assert tensor_based_cfpq(ebnf_to_rsm("S -> S S | a"), graph) == {
        (source, target) for source in graph for target in graph
    }
    assert tensor_based_cfpq(ebnf_to_rsm("S -> S"), graph) == set()


def test_tensor_empty_endpoint_sets_select_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    rsm = ebnf_to_rsm("S -> a*")
    assert tensor_based_cfpq(rsm, graph, set(), set()) == {(10, 10), (10, 20), (20, 20)}
    assert tensor_based_cfpq(rsm, graph, {10}, set()) == {(10, 10), (10, 20)}
    assert tensor_based_cfpq(rsm, graph, set(), {20}) == {(10, 20), (20, 20)}


@pytest.mark.parametrize("text", ["S -> a", "S -> $", "S -> S"])
def test_tensor_empty_graph(text):
    assert tensor_based_cfpq(ebnf_to_rsm(text), MultiDiGraph()) == set()


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_tensor_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        tensor_based_cfpq(ebnf_to_rsm("S -> $"), graph, starts, finals)


def test_tensor_parallel_missing_and_multicharacter_labels():
    graph = MultiDiGraph()
    graph.add_edge(-10, 100)
    graph.add_edge(-10, 100, label=None)
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(100, 200, label="world")
    rsm = ebnf_to_rsm("Root -> hello world")
    assert tensor_based_cfpq(rsm, graph) == {(-10, 200)}


def test_tensor_uses_sparse_boolean_kronecker_products(monkeypatch):
    import project.task3 as task3
    import project.task6 as task6
    import project.task7 as task7

    original_kron = task3.kron
    calls = []

    def checked_kron(left, right, **kwargs):
        assert left.dtype == right.dtype == bool
        result = original_kron(left, right, **kwargs)
        assert isspmatrix_csr(result) and result.dtype == bool
        calls.append(result.shape)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor CFPQ must not delegate to another CFPQ algorithm")

    monkeypatch.setattr(task3, "kron", checked_kron)
    monkeypatch.setattr(task6, "hellings_based_cfpq", forbidden)
    monkeypatch.setattr(task7, "matrix_based_cfpq", forbidden)
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    assert tensor_based_cfpq(ebnf_to_rsm("S -> a b"), graph) == {(10, 30)}
    assert calls


@pytest.mark.parametrize("seed", range(10))
def test_tensor_matches_original_grammar_and_previous_algorithms(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [-10, 50, 100, 1000]
    graph.add_nodes_from(vertices)
    for _ in range(random.randrange(16)):
        graph.add_edge(
            random.choice(vertices),
            random.choice(vertices),
            label=random.choice(["a", "b", None]),
        )
    starts = set(random.sample(vertices, 2))
    finals = set(random.sample(vertices, 2))
    before = list(graph.edges(keys=True, data=True))
    for text in [
        "S -> a S b | $",
        "S -> A B\nA -> a A | $\nB -> b B | $",
        "S -> A\nA -> B\nB -> A | a S b | $",
        "S -> A B C\nA -> a | $\nB -> b | $\nC -> S a | $",
        "S -> S S | a | b",
    ]:
        grammar = CFG.from_text(text)
        productions = set(grammar.productions)
        rsm = cfg_to_rsm(grammar)
        saved_rsm = deepcopy(rsm)
        expected = reference_cfpq(grammar, graph)
        assert tensor_based_cfpq(rsm, graph) == expected
        selected = {(a, b) for a, b in expected if a in starts and b in finals}
        assert tensor_based_cfpq(rsm, graph, starts, finals) == selected
        assert matrix_based_cfpq(grammar, graph, starts, finals) == selected
        assert hellings_based_cfpq(grammar, graph, starts, finals) == selected
        assert rsm == saved_rsm
        assert set(grammar.productions) == productions
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(starts) == 2 and len(finals) == 2
