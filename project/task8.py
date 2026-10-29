"""RSM conversion and tensor-based context-free path querying."""

from networkx import DiGraph
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task6 import cfg_to_weak_normal_form


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build one box per variable from the grammar's weak normal form.

    Each production is a path in its head's box. Variable-labelled transitions
    are calls to other boxes; epsilon productions make the entry state final.
    """
    grammar = cfg_to_weak_normal_form(cfg)
    if grammar.start_symbol is None:
        return RecursiveAutomaton()

    # A terminal may share its value with a variable. Reserve a distinct box
    # label in that case so a literal edge cannot be mistaken for a box call.
    terminal_values = {terminal.value for terminal in grammar.terminals}
    used_labels = terminal_values | {variable.value for variable in grammar.variables}
    call_labels = {}
    counter = 0
    for variable in grammar.variables:
        label = variable.value
        if label in terminal_values:
            while f"__RSM_CALL_{counter}" in used_labels:
                counter += 1
            label = f"__RSM_CALL_{counter}"
            used_labels.add(label)
        call_labels[variable] = Symbol(label)

    automata = {
        variable: NondeterministicFiniteAutomaton() for variable in grammar.variables
    }
    for automaton in automata.values():
        automaton.add_start_state(0)
    for production in grammar.productions:
        automaton = automata[production.head]
        current = 0
        for symbol in production.body:
            target = len(automaton.states)
            label = (
                call_labels[symbol]
                if isinstance(symbol, Variable)
                else Symbol(symbol.value)
            )
            automaton.add_transition(current, label, target)
            current = target
        automaton.add_final_state(current)
    boxes = {
        Box(automaton, call_labels[variable])
        for variable, automaton in automata.items()
    }
    return RecursiveAutomaton(
        initial_label=call_labels[grammar.start_symbol], boxes=boxes
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Parse regular right-hand sides with pyformlang's EBNF/regex syntax.

    S is the entry box when present; otherwise the first production head is used.
    Repeated heads are combined by the library into alternatives of one box.
    """
    heads = [
        line.split("->", 1)[0].strip() for line in ebnf.splitlines() if "->" in line
    ]
    if not heads:
        if ebnf.strip():
            raise ValueError("EBNF must contain a production")
        return RecursiveAutomaton()
    start = "S" if "S" in heads else heads[0]
    return RecursiveAutomaton.from_text(ebnf, start_symbol=Symbol(start))


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Saturate graph relations using sparse Kronecker-product reachability.

    Omitted or empty endpoint sets select all vertices. They only filter the
    answer: intermediate vertices remain available to recursive box calls.
    """
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    size = len(graph_fa.states)
    if size == 0 or rsm.initial_label not in rsm.boxes:
        return set()

    # Local state numbers may repeat between boxes: include the box label in
    # every state identity before reusing the automaton intersection from task 3.
    combined = NondeterministicFiniteAutomaton(
        states={
            (label, state)
            for label, box in rsm.boxes.items()
            for state in box.dfa.states
        }
    )
    for label, box in rsm.boxes.items():
        for source, symbol, target in box.dfa:
            combined.add_transition((label, source), symbol, (label, target))
    rsm_fa = AdjacencyMatrixFA(combined)
    entries = {}
    exits = {}
    for label, box in rsm.boxes.items():
        entries[label] = [
            rsm_fa.state_indices[State((label, state))]
            for state in box.dfa.start_states
        ]
        exits[label] = [
            rsm_fa.state_indices[State((label, state))]
            for state in box.dfa.final_states
        ]

    # A graph edge bearing a box's name is not a completed call. Call relations
    # start empty and are inferred only from paths accepted by the relevant box.
    for label in rsm.boxes:
        graph_fa.matrices[label] = csr_matrix((size, size), dtype=bool)

    changed = True
    while changed:
        changed = False
        product = intersect_automata(rsm_fa, graph_fa)
        closure = product.transitive_closure()
        for label in rsm.boxes:
            relation = graph_fa.matrices[label]
            for start in entries[label]:
                for final in exits[label]:
                    # kron(RSM, graph) numbers (q, v) as q * size + v.
                    # The identity in closure also discovers nullable boxes.
                    relation = (
                        relation
                        + closure[
                            start * size : (start + 1) * size,
                            final * size : (final + 1) * size,
                        ]
                    )
            if relation.nnz != graph_fa.matrices[label].nnz:
                graph_fa.matrices[label] = relation
                changed = True

    result = graph_fa.matrices[rsm.initial_label]
    return {
        (graph_fa.states[source].value, graph_fa.states[target].value)
        for source in graph_fa.start_states
        for target in set(result.getrow(source).indices) & graph_fa.final_states
    }
