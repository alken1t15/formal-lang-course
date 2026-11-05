"""GLL reachability on recursive state machines with a graph-structured stack."""

from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.rsa import RecursiveAutomaton

from project.task2 import graph_to_nfa


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Recognize paths with GLL descriptors, shared calls and cached returns.

    A GSS node (box label, entry vertex) shares one box invocation among all
    callers. Its outgoing edges store (caller node, return state). A descriptor
    (box state, GSS node, graph vertex) represents one suspended parsing step.
    Omitted or empty endpoint sets select all vertices, as in previous tasks.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    if rsm.initial_label not in rsm.boxes:
        return set()

    graph_steps = defaultdict(set)
    for source, symbol, target in graph_nfa:
        graph_steps[source.value, symbol].add(target.value)
    box_steps = {}
    entries = {}
    finals = {}
    for label, box in rsm.boxes.items():
        entries[label] = box.dfa.start_states
        finals[label] = box.dfa.final_states
        transitions = defaultdict(list)
        for source, symbol, target in box.dfa:
            transitions[source].append((symbol, target))
        box_steps[label] = transitions

    agenda = deque()
    descriptors = set()
    gss_edges = defaultdict(set)
    popped = defaultdict(set)

    def add(state, node, vertex):
        descriptor = (state, node, vertex)
        if descriptor not in descriptors:
            descriptors.add(descriptor)
            agenda.append(descriptor)

    for source in graph_nfa.start_states:
        node = (rsm.initial_label, source.value)
        for state in entries[rsm.initial_label]:
            add(state, node, source.value)

    while agenda:
        state, node, vertex = agenda.popleft()
        label, _ = node
        if state in finals[label] and vertex not in popped[node]:
            popped[node].add(vertex)
            # A completed call resumes every continuation currently in the GSS.
            for caller, return_state in tuple(gss_edges[node]):
                add(return_state, caller, vertex)

        # Final states may also have outgoing transitions (e.g. a star in EBNF).
        for symbol, target in box_steps[label].get(state, ()):
            if symbol in rsm.boxes:
                callee = (symbol, vertex)
                edge = (node, target)
                if edge not in gss_edges[callee]:
                    gss_edges[callee].add(edge)
                    # The box may have returned before this caller was found.
                    # Replay cached returns, including zero-length ones.
                    for end in tuple(popped[callee]):
                        add(target, node, end)
                    for entry in entries[symbol]:
                        add(entry, callee, vertex)
            else:
                for end in graph_steps.get((vertex, symbol), ()):
                    add(target, node, end)

    final_vertices = {state.value for state in graph_nfa.final_states}
    return {
        (source.value, target)
        for source in graph_nfa.start_states
        for target in popped.get((rsm.initial_label, source.value), ())
        if target in final_vertices
    }
