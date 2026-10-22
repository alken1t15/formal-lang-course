"""Tests for Hellings queries, including an independent relational oracle."""

from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from pyformlang.cfg import CFG, Terminal

from project.graph_utils import create_two_cycles_graph
from project.task6 import hellings_based_cfpq


def reference_cfpq(grammar, graph):
    """Interpret original productions by relation composition to a fixed point.

    This deliberately does not normalize the grammar or use a Hellings queue.
    """
    relations = {variable: set() for variable in grammar.variables}
    terminal_relations = {}
    for source, target, data in graph.edges(data=True):
        if data.get("label") is not None:
            terminal_relations.setdefault(Terminal(data["label"]), set()).add(
                (source, target)
            )
    while True:
        changed = False
        for production in grammar.productions:
            paths = {(vertex, vertex) for vertex in graph}
            for symbol in production.body:
                steps = (
                    terminal_relations.get(symbol, set())
                    if isinstance(symbol, Terminal)
                    else relations[symbol]
                )
                paths = {
                    (source, target)
                    for source, middle in paths
                    for begin, target in steps
                    if middle == begin
                }
            new_paths = paths - relations[production.head]
            if new_paths:
                relations[production.head].update(new_paths)
                changed = True
        if not changed:
            return relations.get(grammar.start_symbol, set())


def test_hellings_nonregular_balanced_paths_and_endpoint_filters():
    graph = MultiDiGraph()
    graph.add_edges_from(
        (node, node + 1, {"label": label})
        for node, label in enumerate(["a", "a", "b", "b"])
    )
    grammar = CFG.from_text("S -> a S b | $")
    assert hellings_based_cfpq(grammar, graph, {0}, {4}) == {(0, 4)}
    assert hellings_based_cfpq(grammar, graph, {1}, {3}) == {(1, 3)}
    assert hellings_based_cfpq(grammar, graph, {0}, {3}) == set()
    assert hellings_based_cfpq(grammar, graph, {0, 1}, {3, 4}) == {(0, 4), (1, 3)}


def test_hellings_nullable_binary_rules_and_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_edge(-10, 50, label="b")
    graph.add_node(999)
    grammar = CFG.from_text("S -> A B\nA -> $\nB -> b | $")
    assert hellings_based_cfpq(grammar, graph) == {
        (-10, -10),
        (50, 50),
        (999, 999),
        (-10, 50),
    }
    assert hellings_based_cfpq(grammar, graph, {-10}, {50}) == {(-10, 50)}
    assert hellings_based_cfpq(grammar, graph, set(), {999}) == {(999, 999)}


def test_hellings_joins_both_sides_in_a_cycle():
    graph = MultiDiGraph()
    graph.add_edges_from(
        [(10, 20, {"label": "a"}), (20, 30, {"label": "a"}), (30, 10, {"label": "a"})]
    )
    grammar = CFG.from_text("S -> S S | a")
    assert hellings_based_cfpq(grammar, graph) == {(a, b) for a in graph for b in graph}


@pytest.mark.parametrize("text", ["S -> A\nA -> A", "S -> a", "S -> $"])
def test_hellings_empty_graph(text):
    assert hellings_based_cfpq(CFG.from_text(text), MultiDiGraph()) == set()


def test_hellings_empty_grammar():
    graph = MultiDiGraph()
    graph.add_node(10)
    assert hellings_based_cfpq(CFG(), graph) == set()


def test_hellings_unit_cycle_and_long_multicharacter_rules():
    graph = DiGraph()
    graph.add_edge(10, 30, label="hello")
    graph.add_edge(30, 90, label="middle")
    graph.add_edge(90, 100, label="world")
    grammar = CFG.from_text(
        "Root -> A\nA -> Root | hello B world\nB -> middle", start_symbol="Root"
    )
    assert hellings_based_cfpq(grammar, graph) == {(10, 100)}


def test_hellings_parallel_and_unlabeled_edges():
    graph = MultiDiGraph()
    graph.add_edge(10, 20)
    graph.add_edge(10, 20, label=None)
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="b")
    graph.add_edge(20, 30)
    grammar = CFG.from_text("S -> a | b | a a")
    assert hellings_based_cfpq(grammar, graph) == {(10, 20)}


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_hellings_rejects_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        hellings_based_cfpq(CFG.from_text("S -> $"), graph, starts, finals)


def test_hellings_reuses_two_cycles_graph(tmp_path):
    graph = create_two_cycles_graph(2, 3, ("a", "b"), tmp_path / "cycles.dot")
    grammar = CFG.from_text("S -> a S b | $")
    assert hellings_based_cfpq(grammar, graph) == reference_cfpq(grammar, graph)


@pytest.mark.parametrize("seed", range(12))
def test_hellings_matches_original_grammar_fixed_point(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [-10, 50, 90, 1000]
    graph.add_nodes_from(vertices)
    for _ in range(random.randrange(16)):
        graph.add_edge(
            random.choice(vertices),
            random.choice(vertices),
            label=random.choice(["a", "b", None]),
        )
    starts = set(random.sample(vertices, 2))
    finals = set(random.sample(vertices, 2))
    before = list(graph.edges(keys=True, data=True))
    for text in [
        "S -> a S b | $",
        "S -> A B\nA -> a A | $\nB -> b B | $",
        "S -> A\nA -> B\nB -> A | a S b | $",
        "S -> A B C\nA -> a | $\nB -> b | $\nC -> S a | $",
        "S -> S S | a | b",
        "S -> A\nA -> A",
    ]:
        grammar = CFG.from_text(text)
        original = set(grammar.productions)
        expected = reference_cfpq(grammar, graph)
        assert hellings_based_cfpq(grammar, graph) == expected
        assert hellings_based_cfpq(grammar, graph, starts, finals) == {
            (a, b) for a, b in expected if a in starts and b in finals
        }
        assert set(grammar.productions) == original
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(starts) == 2 and len(finals) == 2
