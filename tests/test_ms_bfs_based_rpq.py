"""Tests for multiple-source regular path queries."""

from collections import deque
from random import Random

import pytest
from networkx import MultiDiGraph

from project.task2 import regex_to_dfa
from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def test_bfs_keeps_sources_separate_after_frontiers_merge():
    graph = MultiDiGraph()
    graph.add_edges_from(
        [
            (10, 30, {"label": "a"}),
            (20, 30, {"label": "b"}),
            (30, 40, {"label": "c"}),
            (50, 60, {"label": "a"}),
            (60, 70, {"label": "c"}),
        ]
    )
    assert ms_bfs_based_rpq("(a | b) c", graph, {10, 20, 50}, {40, 70}) == {
        (10, 40),
        (20, 40),
        (50, 70),
    }
    assert ms_bfs_based_rpq("a c", graph, {10, 20, 50}, {40, 70}) == {
        (10, 40),
        (50, 70),
    }


def test_bfs_visits_each_query_state_at_a_vertex():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 20, label="a")
    graph.add_edge(20, 30, label="b")
    # Reaching vertex 20 once must not prevent reading a second "a" there.
    assert ms_bfs_based_rpq("a a b", graph, {10}, {30}) == {(10, 30)}
    assert ms_bfs_based_rpq("a b", graph, {10}, {30}) == {(10, 30)}


@pytest.mark.parametrize(
    "regex, expected",
    [
        ("", set()),
        ("epsilon", {(10, 10), (99, 99)}),
        ("a*", {(10, 10), (10, 20), (99, 99)}),
        ("a", {(10, 20)}),
    ],
)
def test_bfs_empty_word_and_isolated_vertices(regex, expected):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_node(99)
    assert ms_bfs_based_rpq(regex, graph, {10, 99}, {10, 20, 99}) == expected


def test_bfs_empty_endpoint_sets_mean_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    assert ms_bfs_based_rpq("a*", graph, set(), set()) == {(10, 10), (10, 20), (20, 20)}
    assert ms_bfs_based_rpq("a*", graph, {10}, set()) == {(10, 10), (10, 20)}
    assert ms_bfs_based_rpq("a*", graph, set(), {20}) == {(10, 20), (20, 20)}


def test_bfs_parallel_and_unlabeled_edges_and_long_labels():
    graph = MultiDiGraph()
    graph.add_edge(-5, 100)
    graph.add_edge(-5, 100, label=None)
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="hello")
    graph.add_edge(-5, 100, label="other")
    graph.add_edge(100, 200, label="world")
    assert ms_bfs_based_rpq("hello world", graph, {-5}, {200}) == {(-5, 200)}
    assert ms_bfs_based_rpq("h e l l o world", graph, {-5}, {200}) == set()


def test_bfs_empty_graph():
    assert ms_bfs_based_rpq("a*", MultiDiGraph(), set(), set()) == set()


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_bfs_rejects_unknown_vertices(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        ms_bfs_based_rpq("a*", graph, starts, finals)


def reference_query(regex, graph, sources, finals):
    """Search individual vertex/DFA pairs without sparse matrix operations."""
    dfa = regex_to_dfa(regex)
    answer = set()
    for source in sources:
        queue = deque((source, state) for state in dfa.start_states)
        visited = set(queue)
        while queue:
            vertex, state = queue.popleft()
            if vertex in finals and state in dfa.final_states:
                answer.add((source, vertex))
            for _, target, data in graph.out_edges(vertex, data=True):
                if data.get("label") is None:
                    continue
                for next_state in dfa(state, data["label"]):
                    pair = (target, next_state)
                    if pair not in visited:
                        visited.add(pair)
                        queue.append(pair)
    return answer


@pytest.mark.parametrize("seed", range(10))
def test_bfs_matches_independent_search_and_tensor_query(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [-10, 20, 50, 90, 150, 1000]
    graph.add_nodes_from(vertices)
    for _ in range(25):
        graph.add_edge(
            random.choice(vertices),
            random.choice(vertices),
            label=random.choice(["a", "b", "c", None]),
        )
    sources = set(random.sample(vertices, 3))
    finals = set(random.sample(vertices, 2))
    before = list(graph.edges(keys=True, data=True))
    for regex in ["", "epsilon", "a*", "a b", "(a | b)* c", "a (b c)*"]:
        expected = reference_query(regex, graph, sources, finals)
        assert ms_bfs_based_rpq(regex, graph, sources, finals) == expected
        assert tensor_based_rpq(regex, graph, sources, finals) == expected
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(sources) == 3
    assert len(finals) == 2


def test_bfs_long_chain_with_multiple_sources():
    graph = MultiDiGraph()
    graph.add_edges_from((node, node + 1, {"label": "a"}) for node in range(300))
    assert ms_bfs_based_rpq("a a*", graph, {0, 100, 300}, {150, 300}) == {
        (0, 150),
        (0, 300),
        (100, 150),
        (100, 300),
    }


def test_bfs_cycle_terminates_without_accepting_final_state():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")
    graph.add_node(99)
    assert ms_bfs_based_rpq("a* b", graph, {10, 20}, {10, 20, 99}) == set()
    assert ms_bfs_based_rpq("a*", graph, {10, 20}, {10, 20}) == {
        (10, 10),
        (10, 20),
        (20, 10),
        (20, 20),
    }
