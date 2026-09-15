"""Tests for graph utilities from task 1."""

import cfpq_data
import pydot
from networkx import MultiDiGraph

from project.graph_utils import create_two_cycles_graph, get_graph_info


def test_get_graph_info_returns_graph_statistics(monkeypatch):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="b")
    graph.add_edge(1, 2, label="a")
    monkeypatch.setattr(cfpq_data, "download", lambda _: "example.csv")
    monkeypatch.setattr(cfpq_data, "graph_from_csv", lambda _: graph)

    assert get_graph_info("example") == (3, 3, {"a", "b"})


def test_create_two_cycles_graph_writes_dot_file(tmp_path):
    output_path = tmp_path / "two_cycles.dot"

    graph = create_two_cycles_graph(2, 3, ("a", "b"), output_path)
    dot_graphs = pydot.graph_from_dot_file(str(output_path))

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 7
    assert len(dot_graphs) == 1
    assert len(dot_graphs[0].get_edges()) == 7
    assert {edge.get_label().strip('"') for edge in dot_graphs[0].get_edges()} == {
        "a",
        "b",
    }


def test_get_graph_info_returns_empty_labels_for_graph_without_edges(monkeypatch):
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1])
    monkeypatch.setattr(cfpq_data, "download", lambda _: "example.csv")
    monkeypatch.setattr(cfpq_data, "graph_from_csv", lambda _: graph)

    assert get_graph_info("empty") == (2, 0, set())


def test_get_graph_info_deduplicates_repeated_labels(monkeypatch):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    monkeypatch.setattr(cfpq_data, "download", lambda _: "example.csv")
    monkeypatch.setattr(cfpq_data, "graph_from_csv", lambda _: graph)

    assert get_graph_info("repeated-labels") == (3, 2, {"a"})


def test_get_graph_info_uses_downloaded_graph_path(monkeypatch):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="edge")
    calls = []

    def download(name):
        calls.append(("download", name))
        return "example.csv"

    def graph_from_csv(path):
        calls.append(("graph_from_csv", path))
        return graph

    monkeypatch.setattr(cfpq_data, "download", download)
    monkeypatch.setattr(cfpq_data, "graph_from_csv", graph_from_csv)

    assert get_graph_info("example") == (2, 1, {"edge"})
    assert calls == [("download", "example"), ("graph_from_csv", "example.csv")]


def test_two_cycles_graph_has_common_vertex(tmp_path):
    graph = create_two_cycles_graph(2, 3, ("a", "b"), tmp_path / "two_cycles.dot")

    assert graph.in_degree(0) == 2
    assert graph.out_degree(0) == 2


def test_two_cycles_graph_uses_requested_labels(tmp_path):
    graph = create_two_cycles_graph(2, 3, ("left", "right"), tmp_path / "two_cycles.dot")

    assert {edge_data["label"] for _, _, edge_data in graph.edges(data=True)} == {
        "left",
        "right",
    }


def test_two_cycles_graph_creates_dot_file(tmp_path):
    output_path = tmp_path / "two_cycles.dot"

    create_two_cycles_graph(2, 3, ("a", "b"), output_path)

    assert output_path.is_file()


def test_dot_file_contains_all_generated_edges(tmp_path):
    output_path = tmp_path / "two_cycles.dot"
    create_two_cycles_graph(2, 3, ("a", "b"), output_path)

    dot_graph = pydot.graph_from_dot_file(str(output_path))[0]

    assert len(dot_graph.get_edges()) == 7


def test_dot_file_contains_requested_labels(tmp_path):
    output_path = tmp_path / "two_cycles.dot"
    create_two_cycles_graph(2, 3, ("a", "b"), output_path)

    dot_graph = pydot.graph_from_dot_file(str(output_path))[0]

    assert {edge.get_label().strip('"') for edge in dot_graph.get_edges()} == {"a", "b"}
