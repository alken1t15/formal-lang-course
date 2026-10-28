"""Boundary cases for matrix-based context-free path querying."""

import pytest
from networkx import DiGraph, MultiDiGraph
from pyformlang.cfg import CFG

from project.task7 import matrix_based_cfpq


def test_matrix_balanced_paths_with_internal_vertices_outside_filters():
    graph = MultiDiGraph()
    graph.add_edges_from(
        (node, node + 1, {"label": label})
        for node, label in enumerate(["a", "a", "b", "b"])
    )
    grammar = CFG.from_text("S -> a S b | $")
    assert matrix_based_cfpq(grammar, graph, {0}, {4}) == {(0, 4)}
    assert matrix_based_cfpq(grammar, graph, {0, 1}, {3, 4}) == {(0, 4), (1, 3)}
    assert matrix_based_cfpq(grammar, graph, {0}, {3}) == set()


def test_matrix_nullable_variables_and_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_edge(-7, 100, label="b")
    graph.add_node(999)
    grammar = CFG.from_text("S -> A B C\nA -> $\nB -> b | $\nC -> $")
    assert matrix_based_cfpq(grammar, graph) == {
        (-7, -7),
        (100, 100),
        (999, 999),
        (-7, 100),
    }
    assert matrix_based_cfpq(grammar, graph, {-7}, {100}) == {(-7, 100)}


@pytest.mark.parametrize(
    "grammar", [CFG(), CFG.from_text("S -> A\nA -> A"), CFG.from_text("S -> $")]
)
def test_matrix_empty_graph(grammar):
    assert matrix_based_cfpq(grammar, MultiDiGraph()) == set()


def test_matrix_empty_language_on_nonempty_graph():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    for grammar in [CFG(), CFG.from_text("S -> A\nA -> A"), CFG.from_text("S -> b")]:
        assert matrix_based_cfpq(grammar, graph) == set()


def test_matrix_empty_endpoint_sets_select_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    grammar = CFG.from_text("S -> a S | $")
    assert matrix_based_cfpq(grammar, graph, set(), set()) == {
        (10, 10),
        (10, 30),
        (30, 30),
    }
    assert matrix_based_cfpq(grammar, graph, {10}, set()) == {(10, 10), (10, 30)}
    assert matrix_based_cfpq(grammar, graph, set(), {30}) == {(10, 30), (30, 30)}


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_matrix_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        matrix_based_cfpq(CFG.from_text("S -> $"), graph, starts, finals)


def test_matrix_parallel_edges_and_multicharacter_labels():
    graph = MultiDiGraph()
    graph.add_edge(-10, 100)
    graph.add_edge(-10, 100, label=None)
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="other")
    graph.add_edge(100, 300, label="world")
    grammar = CFG.from_text("Root -> A | hello world\nA -> Root", start_symbol="Root")
    assert matrix_based_cfpq(grammar, graph) == {(-10, 300)}


def test_matrix_accepts_digraph_and_distinguishes_variable_like_labels():
    graph = DiGraph()
    graph.add_edge(10, 20, label="S")
    graph.add_edge(20, 30, label="b")
    assert matrix_based_cfpq(CFG.from_text("S -> S | b"), graph) == {(20, 30)}


def test_matrix_cycle_reaches_a_fixed_point():
    graph = MultiDiGraph()
    graph.add_edges_from(
        [(10, 20, {"label": "a"}), (20, 30, {"label": "a"}), (30, 10, {"label": "a"})]
    )
    assert matrix_based_cfpq(CFG.from_text("S -> S S | a"), graph) == {
        (source, target) for source in graph for target in graph
    }
