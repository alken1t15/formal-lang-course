"""GLL tests for shared calls, nullable returns and recursive RSMs."""

from copy import deepcopy
from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from pyformlang.cfg import CFG

from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq
from project.task9 import gll_based_cfpq
from test_hellings_based_cfpq import reference_cfpq


def test_gll_late_caller_reuses_completed_nullable_call():
    graph = MultiDiGraph()
    graph.add_edge(10, 10, label="c")
    graph.add_edge(10, 20, label="b")
    graph.add_edge(10, 30, label="d")
    rsm = ebnf_to_rsm("S -> A b | c A d\nA -> $")
    assert gll_based_cfpq(rsm, graph, {10}, {20, 30}) == {(10, 20), (10, 30)}


def test_gll_shared_callee_has_multiple_return_positions():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    graph.add_edge(20, 40, label="c")
    rsm = ebnf_to_rsm("S -> A b | A c\nA -> a")
    assert gll_based_cfpq(rsm, graph, {10}, {30, 40}) == {(10, 30), (10, 40)}


@pytest.mark.parametrize(
    "text",
    [
        "S -> S a | $",
        "S -> A\nA -> S a | $",
        "S -> S S | a | $",
    ],
)
def test_gll_left_recursion_and_nullable_calls(text):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")
    graph.add_node(99)
    assert gll_based_cfpq(ebnf_to_rsm(text), graph) == {
        (10, 10),
        (10, 20),
        (20, 10),
        (20, 20),
        (99, 99),
    }


def test_gll_recursive_cycle_without_base_case_terminates():
    graph = MultiDiGraph()
    graph.add_edge(10, 10, label="a")
    assert gll_based_cfpq(ebnf_to_rsm("S -> A\nA -> S"), graph) == set()


def test_gll_final_box_state_can_have_outgoing_transitions():
    graph = DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    assert gll_based_cfpq(ebnf_to_rsm("S -> a*"), graph, {0}, {0, 1, 2}) == {
        (0, 0),
        (0, 1),
        (0, 2),
    }


def test_gll_sources_do_not_share_query_progress():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    assert gll_based_cfpq(ebnf_to_rsm("S -> a A\nA -> b"), graph, {0, 1}, {2}) == {
        (0, 2)
    }


def test_gll_balanced_paths_with_internal_vertices_outside_endpoint_sets():
    graph = MultiDiGraph()
    graph.add_edges_from(
        (node, node + 1, {"label": label}) for node, label in enumerate("aabb")
    )
    rsm = ebnf_to_rsm("S -> a S b | $")
    assert gll_based_cfpq(rsm, graph, {0, 1}, {3, 4}) == {(0, 4), (1, 3)}
    assert gll_based_cfpq(rsm, graph, {0}, {3}) == set()


def test_gll_empty_endpoints_select_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(-7, 90, label="a")
    rsm = ebnf_to_rsm("S -> a*")
    assert gll_based_cfpq(rsm, graph, set(), set()) == {(-7, -7), (-7, 90), (90, 90)}
    assert gll_based_cfpq(rsm, graph, {-7}, set()) == {(-7, -7), (-7, 90)}
    assert gll_based_cfpq(rsm, graph, set(), {90}) == {(-7, 90), (90, 90)}


@pytest.mark.parametrize("text", ["S -> a", "S -> $", "S -> S"])
def test_gll_empty_graph(text):
    assert gll_based_cfpq(ebnf_to_rsm(text), MultiDiGraph()) == set()


def test_gll_empty_rsm():
    graph = MultiDiGraph()
    graph.add_node(10)
    assert gll_based_cfpq(ebnf_to_rsm(""), graph) == set()


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_gll_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        gll_based_cfpq(ebnf_to_rsm("S -> $"), graph, starts, finals)


def test_gll_box_named_edges_are_not_completed_calls():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="A")
    graph.add_edge(20, 30, label="b")
    rsm = ebnf_to_rsm("S -> A b\nA -> a")
    assert gll_based_cfpq(rsm, graph) == set()
    graph.add_edge(10, 20, label="a")
    assert gll_based_cfpq(rsm, graph) == {(10, 30)}


def test_gll_parallel_and_missing_edges_and_multicharacter_labels():
    graph = MultiDiGraph()
    graph.add_edge(-10, 100)
    graph.add_edge(-10, 100, label=None)
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(100, 300, label="world")
    rsm = ebnf_to_rsm("Root -> hello A\nA -> world")
    assert gll_based_cfpq(rsm, graph) == {(-10, 300)}


def test_gll_does_not_delegate_to_previous_cfpq_algorithms(monkeypatch):
    import project.task6 as task6
    import project.task7 as task7
    import project.task8 as task8

    def forbidden(*args, **kwargs):
        raise AssertionError("GLL must use descriptors and shared calls")

    monkeypatch.setattr(task6, "hellings_based_cfpq", forbidden)
    monkeypatch.setattr(task7, "matrix_based_cfpq", forbidden)
    monkeypatch.setattr(task8, "tensor_based_cfpq", forbidden)
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    assert gll_based_cfpq(ebnf_to_rsm("S -> A\nA -> a"), graph) == {(10, 20)}


@pytest.mark.parametrize("seed", range(10))
def test_gll_matches_original_grammar_and_tensor_algorithm(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [-7, 90, 300, 1000]
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
        "S -> S S | a | b | $",
        "S -> A B C\nA -> a | $\nB -> b | $\nC -> S a | $",
        "S -> A\nA -> B\nB -> A | a S b | $",
    ]:
        grammar = CFG.from_text(text)
        productions = set(grammar.productions)
        rsm = cfg_to_rsm(grammar)
        saved_rsm = deepcopy(rsm)
        expected = reference_cfpq(grammar, graph)
        assert gll_based_cfpq(rsm, graph) == expected
        selected = {(a, b) for a, b in expected if a in starts and b in finals}
        assert gll_based_cfpq(rsm, graph, starts, finals) == selected
        assert tensor_based_cfpq(rsm, graph, starts, finals) == selected
        assert rsm == saved_rsm
        assert set(grammar.productions) == productions
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(starts) == 2 and len(finals) == 2
