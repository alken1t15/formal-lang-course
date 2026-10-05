"""Multiple-source BFS for regular path queries using sparse matrices."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Find matching paths by advancing all source frontiers simultaneously.

    Empty start or final sets select all graph vertices, as in tasks 2 and 3.
    Each row of the frontier and visited matrices belongs to one graph source.
    """
    query = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    product = intersect_automata(query, graph_fa)
    if not product.start_states or not product.final_states:
        return set()

    graph_size = len(graph_fa.states)
    sources = list(graph_fa.start_states)
    source_rows = {state: row for row, state in enumerate(sources)}
    rows, columns = [], []
    for start in product.start_states:
        rows.append(source_rows[start % graph_size])
        columns.append(start)
    shape = (len(sources), len(product.states))
    frontier = csr_matrix(
        ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
    )
    visited = frontier.copy()

    # Product transitions already require the graph and query to read the same
    # symbol. Their union is therefore sufficient for the BFS step.
    adjacency = csr_matrix((len(product.states), len(product.states)), dtype=bool)
    for matrix in product.matrices.values():
        adjacency = adjacency + matrix

    while frontier.nnz:
        reached = frontier @ adjacency
        # The overlap is a subset of reached: inequality removes visited pairs
        # without constructing a dense complement of the visited matrix.
        frontier = reached != reached.multiply(visited)
        visited = visited + frontier

    answer = set()
    for row, source in enumerate(sources):
        # Initial pairs are visited too, so an accepting query start state
        # accounts for a path of length zero.
        for final in set(visited.getrow(row).indices) & product.final_states:
            answer.add(
                (
                    graph_fa.states[source].value,
                    graph_fa.states[final % graph_size].value,
                )
            )
    return answer
