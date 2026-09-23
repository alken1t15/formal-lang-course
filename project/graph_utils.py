"""Utilities for loading graph data and creating synthetic graphs."""

from pathlib import Path

import cfpq_data
import pydot
from networkx import MultiDiGraph
from cfpq_data.graphs.generators import labeled_two_cycles_graph


def get_graph_info(graph_name: str) -> tuple[int, int, set[str]]:
    """Return the number of nodes, edges, and edge labels of a dataset graph.

    The graph is downloaded from the CFPQ_Data dataset and parsed from CSV.
    """
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    labels = {
        edge_data["label"]
        for _, _, edge_data in graph.edges(data=True)
        if "label" in edge_data
    }
    return graph.number_of_nodes(), graph.number_of_edges(), labels


def create_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> MultiDiGraph:
    """Create a labeled two-cycles graph and save it as a DOT file.

    The cycle sizes are passed unchanged to CFPQ_Data's
    ``labeled_two_cycles_graph`` generator. Its convention is that each size
    excludes the common vertex of the two cycles.
    """
    graph = labeled_two_cycles_graph(
        first_cycle_nodes,
        second_cycle_nodes,
        labels=labels,
    )

    dot_graph = pydot.Dot(graph_type="digraph")
    for node in graph.nodes:
        dot_graph.add_node(pydot.Node(str(node)))
    for source, target, edge_data in graph.edges(data=True):
        dot_graph.add_edge(
            pydot.Edge(str(source), str(target), label=str(edge_data["label"]))
        )
    dot_graph.write_raw(str(output_path))

    return graph
