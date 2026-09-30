"""Tests for tensor-based regular path queries."""

from collections import deque
from random import Random

import pytest
from networkx import MultiDiGraph

from project.graph_utils import create_two_cycles_graph
from project.task2 import regex_to_dfa
from project.task3 import tensor_based_rpq


def reference_rpq(regex, graph, starts, finals):
    """Explore pairs of a vertex and a DFA state without matrix operations."""
    dfa = regex_to_dfa(regex)
    answer = set()
    for source in starts:
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


def test_rpq_preserves_vertex_numbers_and_filters_endpoints():
    graph = MultiDiGraph()
    graph.add_nodes_from([10, 30, 50, 70])
    graph.add_edges_from([(10, 30, {"label": "a"}), (30, 50, {"label": "b"})])
    graph.add_edge(10, 30, label="c")
    assert tensor_based_rpq("a b", graph, {10}, {50, 70}) == {(10, 50)}
    assert tensor_based_rpq("a b", graph, {30}, {50}) == set()
    assert tensor_based_rpq("c", graph, {10}, {30}) == {(10, 30)}
    assert tensor_based_rpq("a b", graph, {10}, {30}) == set()


def test_rpq_handles_zero_length_paths_and_isolated_vertices():
    graph = MultiDiGraph()
    graph.add_nodes_from([10, 30, 50])
    graph.add_edge(10, 30, label="a")
    assert tensor_based_rpq("a*", graph, {10, 50}, {30, 50}) == {(10, 30), (50, 50)}
    assert tensor_based_rpq("epsilon", graph, set(graph), set(graph)) == {
        (10, 10),
        (30, 30),
        (50, 50),
    }
    assert tensor_based_rpq("a", graph, {50}, {50}) == set()


def test_rpq_empty_endpoint_sets_select_all_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    assert tensor_based_rpq("a*", graph, set(), set()) == {(10, 10), (10, 30), (30, 30)}
    assert tensor_based_rpq("a*", graph, {10}, set()) == {(10, 10), (10, 30)}
    assert tensor_based_rpq("a*", graph, set(), {30}) == {(10, 30), (30, 30)}


def test_rpq_ignores_missing_labels_and_keeps_parallel_edges():
    graph = MultiDiGraph()
    graph.add_edge(10, 30)
    graph.add_edge(10, 30, label=None)
    graph.add_edge(10, 30, label="a")
    graph.add_edge(10, 30, label="b")
    graph.add_edge(30, 50)
    assert tensor_based_rpq("a | b", graph, {10}, {30, 50}) == {(10, 30)}
    assert tensor_based_rpq("a a", graph, {10}, {50}) == set()


def test_rpq_empty_graph():
    assert tensor_based_rpq("a*", MultiDiGraph(), set(), set()) == set()


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_rpq_rejects_vertices_outside_graph(starts, finals):
    graph = MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError):
        tensor_based_rpq("a*", graph, starts, finals)


def test_rpq_uses_generated_graph_from_task1(tmp_path):
    graph = create_two_cycles_graph(2, 3, ("a", "b"), tmp_path / "cycles.dot")
    starts, finals = set(graph), set(graph)
    assert tensor_based_rpq("a* b*", graph, starts, finals) == reference_rpq(
        "a* b*", graph, starts, finals
    )


@pytest.mark.parametrize("seed", range(10))
def test_rpq_matches_independent_graph_search(seed):
    random = Random(seed)
    graph = MultiDiGraph()
    vertices = [10, 30, 50, 70, 90]
    graph.add_nodes_from(vertices)
    for _ in range(15):
        graph.add_edge(
            random.choice(vertices),
            random.choice(vertices),
            label=random.choice(["a", "b", "c"]),
        )
    starts = set(random.sample(vertices, 2))
    finals = set(random.sample(vertices, 3))
    before = list(graph.edges(keys=True, data=True))
    for regex in ["a", "a b", "a*", "(a | b)* c", "epsilon", "a b | c a"]:
        assert tensor_based_rpq(regex, graph, starts, finals) == reference_rpq(
            regex, graph, starts, finals
        )
    assert list(graph.edges(keys=True, data=True)) == before
    assert len(starts) == 2
    assert len(finals) == 3


def test_rpq_is_unchanged_by_vertex_renaming():
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "b"}),
            (2, 0, {"label": "a"}),
            (0, 2, {"label": "b"}),
        ]
    )
    renaming = {0: 900, 1: -7, 2: 42, 3: 10000}
    renamed = MultiDiGraph()
    # A different insertion order must not affect the index correspondence.
    renamed.add_nodes_from(renaming[node] for node in reversed(list(graph)))
    for source, target, data in graph.edges(data=True):
        renamed.add_edge(renaming[source], renaming[target], **data)
    starts, finals = {0, 3}, {0, 2, 3}
    for regex in ["a b", "(a | b)*", "epsilon"]:
        expected = {
            (renaming[source], renaming[target])
            for source, target in tensor_based_rpq(regex, graph, starts, finals)
        }
        assert (
            tensor_based_rpq(
                regex,
                renamed,
                {renaming[node] for node in starts},
                {renaming[node] for node in finals},
            )
            == expected
        )


def test_rpq_requires_labels_in_the_requested_order():
    graph = MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "b"}),
            (1, 2, {"label": "a"}),
            (0, 3, {"label": "a"}),
            (3, 4, {"label": "b"}),
        ]
    )
    # Both final vertices are reachable, but only one path spells "a b".
    assert tensor_based_rpq("a b", graph, {0}, {2, 4}) == {(0, 4)}
    assert tensor_based_rpq("b a", graph, {0}, {2, 4}) == {(0, 2)}
