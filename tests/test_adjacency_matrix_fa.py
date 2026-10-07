"""Tests for sparse automata and their intersection."""

from itertools import product

import pytest
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import isspmatrix_csr

from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


@pytest.mark.parametrize("regex", ["", "epsilon", "a", "a*", "a b | c", "(a | b)* a"])
def test_matrix_automaton_matches_dfa(regex):
    dfa = regex_to_dfa(regex)
    matrix_automaton = AdjacencyMatrixFA(dfa)
    for length in range(5):
        for word in product("abc", repeat=length):
            assert matrix_automaton.accepts(iter(word)) == dfa.accepts(word)
    assert matrix_automaton.is_empty() == dfa.is_empty()


def test_matrix_automaton_preserves_nondeterminism_and_isolated_states():
    nfa = NondeterministicFiniteAutomaton(states={State("isolated")})
    nfa.add_start_state("source")
    nfa.add_start_state(100)
    nfa.add_final_state("target")
    nfa.add_transitions(
        [
            ("source", "a", "dead"),
            ("source", "a", "middle"),
            ("middle", "b", "target"),
            (100, "c", "target"),
        ]
    )
    before = nfa.to_dict()
    automaton = AdjacencyMatrixFA(nfa)
    assert set(automaton.states) == nfa.states
    assert automaton.accepts([Symbol("a"), Symbol("b")])
    assert automaton.accepts(["c"])
    assert not automaton.accepts(["a"])
    assert not automaton.accepts(["unknown"])
    assert not automaton.is_empty()
    assert nfa.to_dict() == before
    for matrix in automaton.matrices.values():
        assert isspmatrix_csr(matrix)
        assert matrix.dtype == bool
        assert matrix.shape == (len(nfa.states), len(nfa.states))


@pytest.mark.parametrize(
    "start, final, expected_empty",
    [
        (None, None, True),
        (10, None, True),
        (None, 10, True),
        (10, 20, True),
        (10, 10, False),
    ],
)
def test_empty_language_and_empty_word(start, final, expected_empty):
    nfa = NondeterministicFiniteAutomaton(states={State(10), State(20)})
    if start is not None:
        nfa.add_start_state(start)
    if final is not None:
        nfa.add_final_state(final)
    automaton = AdjacencyMatrixFA(nfa)
    assert automaton.is_empty() is expected_empty
    assert automaton.accepts([]) is (not expected_empty)


def test_empty_automaton():
    automaton = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())
    assert automaton.is_empty()
    assert not automaton.accepts([])
    assert not automaton.accepts(["a"])
    assert automaton.transitive_closure().shape == (0, 0)


def test_transitive_closure_includes_long_paths_and_cycles():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_start_state(0)
    nfa.add_final_state(12)
    nfa.add_transitions((index, "a", index + 1) for index in range(12))
    nfa.add_transition(12, "b", 5)
    automaton = AdjacencyMatrixFA(nfa)
    closure = automaton.transitive_closure()
    indices = automaton.state_indices
    assert closure[indices[State(0)], indices[State(12)]]
    assert closure[indices[State(12)], indices[State(5)]]
    assert not closure[indices[State(12)], indices[State(0)]]
    assert closure.diagonal().all()
    assert not automaton.is_empty()


@pytest.mark.parametrize(
    "first, second",
    [
        ("", "a*"),
        ("epsilon", "a*"),
        ("a*", "b*"),
        ("a", "b"),
        ("(a | b)*", "a b*"),
        ("a b | c", "a* b*"),
    ],
)
def test_intersection_matches_both_languages(first, second):
    dfa1, dfa2 = regex_to_dfa(first), regex_to_dfa(second)
    intersection = intersect_automata(AdjacencyMatrixFA(dfa1), AdjacencyMatrixFA(dfa2))
    for length in range(5):
        for word in product("abc", repeat=length):
            assert intersection.accepts(word) == (
                dfa1.accepts(word) and dfa2.accepts(word)
            )
    assert intersection.is_empty() == dfa1.get_intersection(dfa2).is_empty()


def test_intersection_of_nondeterministic_automata():
    first = NondeterministicFiniteAutomaton()
    first.add_start_state("s")
    first.add_final_state("f")
    first.add_transitions([("s", "a", "f"), ("s", "a", "other")])
    second = NondeterministicFiniteAutomaton()
    second.add_start_state(10)
    second.add_start_state(20)
    second.add_final_state(30)
    second.add_transition(20, "a", 30)
    intersection = intersect_automata(
        AdjacencyMatrixFA(first), AdjacencyMatrixFA(second)
    )
    assert intersection.accepts(["a"])
    assert not intersection.accepts([])
    assert not intersection.accepts(["a", "a"])
    assert not intersection.is_empty()


def test_intersection_with_empty_automaton():
    empty = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())
    nonempty = AdjacencyMatrixFA(regex_to_dfa("a*"))
    for first, second in [(empty, nonempty), (nonempty, empty)]:
        intersection = intersect_automata(first, second)
        assert intersection.is_empty()
        assert not intersection.accepts([])


def test_intersection_with_multiple_start_and_final_states():
    first = NondeterministicFiniteAutomaton()
    first.add_start_state("left")
    first.add_start_state("right")
    first.add_final_state("left_final")
    first.add_final_state("right_final")
    first.add_transitions(
        [
            ("left", "a", "left_final"),
            ("right", "b", "right_final"),
        ]
    )
    second = NondeterministicFiniteAutomaton()
    second.add_start_state(10)
    second.add_start_state(20)
    second.add_final_state(30)
    second.add_final_state(40)
    second.add_transitions([(10, "b", 30), (20, "a", 40)])
    intersection = intersect_automata(
        AdjacencyMatrixFA(first), AdjacencyMatrixFA(second)
    )
    assert len(intersection.start_states) == 4
    assert len(intersection.final_states) == 4
    for word in [[], ["a"], ["b"], ["c"], ["a", "b"]]:
        assert intersection.accepts(word) == (
            first.accepts(word) and second.accepts(word)
        )
    assert not intersection.is_empty()
