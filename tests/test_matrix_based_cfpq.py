"""Boundary cases for matrix-based context-free path querying."""

from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from pyformlang.cfg import CFG

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq
from test_hellings_based_cfpq import reference_cfpq


def test_matrix_balanced_paths_with_internal_vertices_outside_filters():
    graph = MultiDiGraph()
    graph.add_edges_from(
        (node, node + 1, {"label": label})
        for node, label in enumerate(["a", "a", "b", "b"])
    )
    grammar = CFG.from_text("S -> a S b | $")
    assert matrix_based_cfpq(grammar, graph, {0}, {4}) == {(0, 4)}
    assert matrix_based_cfpq(grammar, graph, {0, 1}, {3, 4}) == {(0, 4), (1, 3)}
    assert matrix_based_cfpq(grammar, graph, {0}, {3}) == set()


def test_matrix_nullable_variables_and_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_edge(-7, 100, label="b")
    graph.add_node(999)
    grammar = CFG.from_text("S -> A B C\nA -> $\nB -> b | $\nC -> $")
    assert matrix_based_cfpq(grammar, graph) == {
        (-7, -7),
        (100, 100),
        (999, 999),
        (-7, 100),
    }
    assert matrix_based_cfpq(grammar, graph, {-7}, {100}) == {(-7, 100)}


@pytest.mark.parametrize(
    "grammar", [CFG(), CFG.from_text("S -> A\nA -> A"), CFG.from_text("S -> $")]
)
def test_matrix_empty_graph(grammar):
    assert matrix_based_cfpq(grammar, MultiDiGraph()) == set()


def test_matrix_empty_language_on_nonempty_graph():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    for grammar in [CFG(), CFG.from_text("S -> A\nA -> A"), CFG.from_text("S -> b")]:
        assert matrix_based_cfpq(grammar, graph) == set()


def test_matrix_empty_endpoint_sets_select_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    grammar = CFG.from_text("S -> a S | $")
    assert matrix_based_cfpq(grammar, graph, set(), set()) == {
        (10, 10),
        (10, 30),
        (30, 30),
    }
    assert matrix_based_cfpq(grammar, graph, {10}, set()) == {(10, 10), (10, 30)}
    assert matrix_based_cfpq(grammar, graph, set(), {30}) == {(10, 30), (30, 30)}


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_matrix_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        matrix_based_cfpq(CFG.from_text("S -> $"), graph, starts, finals)


def test_matrix_parallel_edges_and_multicharacter_labels():
    graph = MultiDiGraph()
    graph.add_edge(-10, 100)
    graph.add_edge(-10, 100, label=None)
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="hello")
    graph.add_edge(-10, 100, label="other")
    graph.add_edge(100, 300, label="world")
    grammar = CFG.from_text("Root -> A | hello world\nA -> Root", start_symbol="Root")
    assert matrix_based_cfpq(grammar, graph) == {(-10, 300)}


def test_matrix_accepts_digraph_and_distinguishes_variable_like_labels():
    graph = DiGraph()
    graph.add_edge(10, 20, label="S")
    graph.add_edge(20, 30, label="b")
    assert matrix_based_cfpq(CFG.from_text("S -> S | b"), graph) == {(20, 30)}


def test_matrix_cycle_reaches_a_fixed_point():
    graph = MultiDiGraph()
    graph.add_edges_from(
        [(10, 20, {"label": "a"}), (20, 30, {"label": "a"}), (30, 10, {"label": "a"})]
    )
    assert matrix_based_cfpq(CFG.from_text("S -> S S | a"), graph) == {
        (source, target) for source in graph for target in graph
    }


@pytest.mark.parametrize("seed", range(12))
def test_matrix_matches_hellings_and_original_grammar_relations(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [-10, 50, 100, 300, 1000]
    graph.add_nodes_from(vertices)
    for _ in range(random.randrange(20)):
        graph.add_edge(
            random.choice(vertices),
            random.choice(vertices),
            label=random.choice(["a", "b", None]),
        )
    starts = set(random.sample(vertices, 2))
    finals = set(random.sample(vertices, 3))
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
        productions = set(grammar.productions)
        # Reuse the independent interpretation of original rules from task 6 tests.
        expected = reference_cfpq(grammar, graph)
        assert matrix_based_cfpq(grammar, graph) == expected
        selected = {(a, b) for a, b in expected if a in starts and b in finals}
        assert matrix_based_cfpq(grammar, graph, starts, finals) == selected
        assert hellings_based_cfpq(grammar, graph, starts, finals) == selected
        assert set(grammar.productions) == productions
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(starts) == 2 and len(finals) == 3


def test_matrix_uses_boolean_sparse_multiplication(monkeypatch):
    import project.task6 as task6
    from scipy.sparse import csr_matrix

    multiply = csr_matrix.__matmul__
    calls = []

    def checked_multiply(left, right):
        assert left.dtype == bool and right.dtype == bool
        result = multiply(left, right)
        assert result.dtype == bool
        calls.append(result.shape)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Matrix CFPQ must use sparse operations, not Hellings or dense arrays"
        )

    monkeypatch.setattr(csr_matrix, "__matmul__", checked_multiply)
    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(task6, "hellings_based_cfpq", forbidden)
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph) == {(10, 30)}
    assert calls and all(shape == (3, 3) for shape in calls)


def test_matrix_many_paths_do_not_overflow_a_path_counter():
    graph = MultiDiGraph()
    for middle in range(1, 301):
        graph.add_edge(0, middle, label="a")
        graph.add_edge(middle, 1000, label="b")
    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph, {0}, {1000}) == {
        (0, 1000)
    }


def test_matrix_propagates_reachability_through_a_long_chain():
    graph = MultiDiGraph()
    graph.add_edges_from((node, node + 1, {"label": "a"}) for node in range(60))
    grammar = CFG.from_text("S -> S S | a")
    assert matrix_based_cfpq(grammar, graph, {0, 30, 60}, {30, 60}) == {
        (0, 30),
        (0, 60),
        (30, 60),
    }
