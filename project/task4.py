"""Multiple-source BFS for regular path queries using sparse matrices."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Advance source frontiers on graph matrices without building an intersection.

    Empty start or final sets select all graph vertices, as in tasks 2 and 3.
    For each query state, a frontier matrix has one row per source and one column
    per graph vertex. All state indices used below are integers.
    """
    query = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    if not graph_fa.start_states or not graph_fa.final_states or not query.final_states:
        return set()

    sources: list[int] = list(graph_fa.start_states)
    shape = (len(sources), len(graph_fa.states))
    initial = csr_matrix(
        ([True] * len(sources), (list(range(len(sources))), sources)),
        shape=shape,
        dtype=bool,
    )
    frontier = [csr_matrix(shape, dtype=bool) for _ in query.states]
    for state in query.start_states:
        frontier[state] = initial.copy()
    visited = [matrix.copy() for matrix in frontier]

    # Store only query transitions whose symbols occur in the graph.
    # No Kronecker product or adjacency matrix of the intersection is needed.
    transitions = []
    for symbol in query.matrices.keys() & graph_fa.matrices.keys():
        rows, columns = query.matrices[symbol].nonzero()
        transitions.extend(
            (source, target, symbol) for source, target in zip(rows, columns)
        )

    while any(matrix.nnz for matrix in frontier):
        reached = [csr_matrix(shape, dtype=bool) for _ in query.states]
        for source, target, symbol in transitions:
            if frontier[source].nnz:
                # Read the symbol on graph edges and move to the next query state.
                reached[target] = (
                    reached[target] + frontier[source] @ graph_fa.matrices[symbol]
                )
        for state in range(len(query.states)):
            # Remove visited pairs independently for each source and query state.
            frontier[state] = reached[state] != reached[state].multiply(visited[state])
            visited[state] = visited[state] + frontier[state]

    accepted = csr_matrix(shape, dtype=bool)
    for state in query.final_states:
        accepted = accepted + visited[state]
    answer = set()
    for row, source in enumerate(sources):
        # Initial frontiers also account for accepted words of length zero.
        for target in set(accepted.getrow(row).indices) & graph_fa.final_states:
            answer.add((graph_fa.states[source].value, graph_fa.states[target].value))
    return answer
