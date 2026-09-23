"""Tests for finite automata construction from task 2."""

from networkx import MultiDiGraph

from project.task2 import graph_to_nfa, regex_to_dfa


def test_regex_to_dfa_accepts_the_regular_language():
    automaton = regex_to_dfa("a.b")

    assert automaton.accepts(["a", "b"])
    assert not automaton.accepts(["a"])
    assert not automaton.accepts(["b", "a"])


def test_regex_to_dfa_returns_minimal_deterministic_automaton():
    automaton = regex_to_dfa("a|a.b")

    assert automaton.is_deterministic()
    assert len(automaton.states) == len(automaton.minimize().states)


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


def test_graph_to_nfa_keeps_parallel_edges():
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")

    automaton = graph_to_nfa(graph, {0}, {1})

    assert automaton.accepts(["a"])
    assert automaton.accepts(["b"])


def test_graph_to_nfa_ignores_unlabeled_edges():
    graph = MultiDiGraph()
    graph.add_edge(0, 1)

    automaton = graph_to_nfa(graph, {0}, {1})

    assert automaton.is_empty()
