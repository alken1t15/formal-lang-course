"""Sparse-matrix automata and tensor-based regular path queries."""

from collections.abc import Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import csr_matrix, eye, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """Represent transitions by a Boolean CSR matrix for each symbol.

    Matrix rows and columns use consecutive indices rather than state values.
    ``states`` and ``state_indices`` preserve the correspondence with the input.
    """

    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        # Include isolated states too; transitions alone do not list them.
        self.states = tuple(automaton.states) if automaton is not None else ()
        self.state_indices = {state: index for index, state in enumerate(self.states)}
        self.start_states = set()
        self.final_states = set()
        self.matrices: dict[Symbol, csr_matrix] = {}
        if automaton is None:
            return

        self.start_states = {
            self.state_indices[state] for state in automaton.start_states
        }
        self.final_states = {
            self.state_indices[state] for state in automaton.final_states
        }
        transitions = {}
        for source, by_symbol in automaton.to_dict().items():
            for symbol, targets in by_symbol.items():
                # DFA transitions have one target, NFA transitions have a set.
                if isinstance(targets, State):
                    targets = {targets}
                rows, columns = transitions.setdefault(symbol, ([], []))
                for target in targets:
                    rows.append(self.state_indices[source])
                    columns.append(self.state_indices[target])
        for symbol, (rows, columns) in transitions.items():
            self.matrices[symbol] = csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(len(self.states), len(self.states)),
                dtype=bool,
            )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Read a word by multiplying a vector of active states by matrices."""
        active = csr_matrix(
            (
                [True] * len(self.start_states),
                ([0] * len(self.start_states), list(self.start_states)),
            ),
            shape=(1, len(self.states)),
            dtype=bool,
        )
        for symbol in word:
            matrix = self.matrices.get(symbol)
            if matrix is None:
                return False
            # Boolean multiplication keeps every possible NFA branch active.
            active = active @ matrix
            if active.nnz == 0:
                return False
        return bool(set(active.indices) & self.final_states)

    def transitive_closure(self) -> csr_matrix:
        """Return reachability, including paths of length zero."""
        size = len(self.states)
        # The diagonal includes empty paths, needed for the empty word.
        reachable = eye(size, format="csr", dtype=bool)
        # Reachability ignores labels, so combine transitions of all symbols.
        for matrix in self.matrices.values():
            reachable = reachable + matrix
        # Each squaring doubles the maximum path length represented so far.
        while True:
            expanded = reachable + reachable @ reachable
            # Entries only change from False to True: unchanged nnz means
            # no new reachable pairs were added.
            if expanded.nnz == reachable.nnz:
                return expanded
            reachable = expanded

    def is_empty(self) -> bool:
        """Check whether any final state is reachable from a start state."""
        if not self.start_states or not self.final_states:
            return True
        reachable = self.transitive_closure()
        return (
            reachable[list(self.start_states), :][:, list(self.final_states)].nnz == 0
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Intersect languages using Kronecker products of matching transitions."""
    result = AdjacencyMatrixFA()
    result.states = tuple(
        (first, second) for first in automaton1.states for second in automaton2.states
    )
    result.state_indices = {state: index for index, state in enumerate(result.states)}
    second_size = len(automaton2.states)
    # kron(A, B) numbers a pair (i, j) as i * len(B) + j.
    result.start_states = {
        first * second_size + second
        for first in automaton1.start_states
        for second in automaton2.start_states
    }
    result.final_states = {
        first * second_size + second
        for first in automaton1.final_states
        for second in automaton2.final_states
    }
    # Both automata must read the same symbol in each product transition.
    for symbol in automaton1.matrices.keys() & automaton2.matrices.keys():
        result.matrices[symbol] = kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        )
    return result


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Find vertex pairs connected by a path whose labels match ``regex``.

    As in task 2, empty start or final sets select all graph vertices.
    """
    query = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    product = intersect_automata(query, graph_automaton)
    reachable = product.transitive_closure()
    graph_size = len(graph_automaton.states)
    answer = set()
    for start in product.start_states:
        row = reachable.getrow(start)
        for final in set(row.indices) & product.final_states:
            # Graph states are the second component of the product; modulo
            # recovers their indices even when vertex numbers are not consecutive.
            source = graph_automaton.states[start % graph_size].value
            target = graph_automaton.states[final % graph_size].value
            answer.add((source, target))
    return answer
