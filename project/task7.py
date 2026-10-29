"""Context-free path querying by Boolean sparse matrix multiplication."""

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_matrix, eye

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Compute grammar-constrained reachability to a matrix fixed point.

    Matrix A contains (u, v) exactly when the variable A derives a path from
    u to v. Omitted or empty endpoint sets select all vertices, as before.
    """
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    grammar = cfg_to_weak_normal_form(cfg)
    size = len(graph_fa.states)
    if size == 0:
        return set()

    matrices = {
        variable: csr_matrix((size, size), dtype=bool) for variable in grammar.variables
    }
    binary_rules = []
    for production in grammar.productions:
        head, body = production.head, production.body
        if not body:
            # Every vertex, including an isolated one, has an empty path to itself.
            matrices[head] = matrices[head] + eye(size, format="csr", dtype=bool)
        elif len(body) == 1:
            # CFG terminals and automaton symbols have different wrapper types.
            edges = graph_fa.matrices.get(Symbol(body[0].value))
            if edges is not None:
                matrices[head] = matrices[head] + edges
        else:
            binary_rules.append((head, body[0], body[1]))

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_rules:
            # Boolean multiplication composes paths, addition combines alternatives.
            expanded = matrices[head] + matrices[left] @ matrices[right]
            # Reachable pairs only accumulate, so unchanged nnz means no new pairs.
            if expanded.nnz != matrices[head].nnz:
                matrices[head] = expanded
                changed = True

    result = matrices.get(grammar.start_symbol)
    if result is None:
        return set()
    answer = set()
    # Endpoint filters must not restrict intermediate vertices during closure.
    for source in graph_fa.start_states:
        for target in set(result.getrow(source).indices) & graph_fa.final_states:
            answer.add((graph_fa.states[source].value, graph_fa.states[target].value))
    return answer
