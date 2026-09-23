"""Finite automata construction for task 2."""

from collections.abc import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal deterministic finite automaton for ``regex``."""
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: Set[int] | None = None,
    final_states: Set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    """Convert a labeled directed multigraph into a nondeterministic FA.

    Omitted or empty ``start_states`` or ``final_states`` means every graph vertex
    is respectively a start or final state.
    """
    graph_states = set(graph.nodes)
    starts = graph_states if not start_states else set(start_states)
    finals = graph_states if not final_states else set(final_states)

    if not starts <= graph_states:
        raise ValueError("start_states contains vertices outside the graph")
    if not finals <= graph_states:
        raise ValueError("final_states contains vertices outside the graph")
    automaton = NondeterministicFiniteAutomaton(
        states={State(state) for state in graph_states}
    )
    for state in starts:
        automaton.add_start_state(State(state))
    for state in finals:
        automaton.add_final_state(State(state))

    for source, target, edge_data in graph.edges(data=True):
        if "label" in edge_data:
            automaton.add_transition(
                State(source), Symbol(edge_data["label"]), State(target)
            )

    return automaton
