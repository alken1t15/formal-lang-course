"""Tests for finite automata construction from task 2."""

import cfpq_data
import pytest
from networkx import MultiDiGraph

from project.graph_utils import create_two_cycles_graph
from project.task2 import graph_to_nfa, regex_to_dfa


def test_regex_to_dfa_accepts_the_regular_language():
    automaton = regex_to_dfa("a.b")

    assert automaton.accepts(["a", "b"])
    assert not automaton.accepts(["a"])
    assert not automaton.accepts(["b", "a"])


def test_regex_to_dfa_returns_minimal_deterministic_automaton():
    automaton = regex_to_dfa("a|b")

    assert automaton.is_deterministic()
    assert len(automaton.states) == 2
    assert automaton.accepts(["a"])
    assert automaton.accepts(["b"])


def test_regex_to_dfa_supports_kleene_star():
    automaton = regex_to_dfa("a*")

    assert automaton.accepts([])
    assert automaton.accepts(["a", "a", "a"])
    assert not automaton.accepts(["b"])


def test_graph_to_nfa_uses_explicit_start_and_final_states():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    automaton = graph_to_nfa(graph, {0}, {2})

    assert automaton.accepts(["a", "b"])
    assert not automaton.accepts(["a"])
    assert not automaton.accepts(["b"])


def test_graph_to_nfa_uses_all_vertices_when_states_are_empty():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    automaton = graph_to_nfa(graph, set(), set())

    assert automaton.accepts([])
    assert automaton.accepts(["a"])


def test_graph_to_nfa_defaults_to_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    automaton = graph_to_nfa(graph)

    assert automaton.accepts([])
    assert automaton.accepts(["a"])


def test_graph_to_nfa_keeps_parallel_edges():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")

    automaton = graph_to_nfa(graph, {0}, {1})

    assert automaton.accepts(["a"])
    assert automaton.accepts(["b"])


def test_graph_to_nfa_preserves_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1, 7])
    graph.add_edge(0, 1, label="a")

    automaton = graph_to_nfa(graph, {0}, {1})

    assert {state.value for state in automaton.states} == set(graph.nodes)
    assert automaton.accepts(["a"])


def test_graph_to_nfa_rejects_vertices_outside_graph():
    graph = MultiDiGraph()
    graph.add_node(0)

    with pytest.raises(ValueError, match="start_states"):
        graph_to_nfa(graph, {1}, {0})
    with pytest.raises(ValueError, match="final_states"):
        graph_to_nfa(graph, {0}, {1})


@pytest.mark.parametrize("edge_data", [{}, {"label": None}])
def test_graph_to_nfa_ignores_unlabeled_edges(edge_data):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, **edge_data)

    automaton = graph_to_nfa(graph, {0}, {1})

    assert automaton.is_empty()


def test_graph_to_nfa_accepts_cycles_from_task_1(tmp_path):
    graph = create_two_cycles_graph(2, 3, ("a", "b"), tmp_path / "cycles.dot")

    automaton = graph_to_nfa(graph, {0}, {0})

    assert automaton.accepts(["a", "a", "a"])
    assert automaton.accepts(["b", "b", "b", "b"])
    assert not automaton.accepts(["a", "b"])


def test_graph_to_nfa_accepts_graph_loaded_from_cfpq_data_csv(tmp_path):
    graph_path = tmp_path / "graph.csv"
    graph_path.write_text("0 1 subClassOf\n1 2 type\n")
    graph = cfpq_data.graph_from_csv(graph_path)

    automaton = graph_to_nfa(graph, {0}, {2})

    assert automaton.accepts(["subClassOf", "type"])
    assert not automaton.accepts(["type"])
