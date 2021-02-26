from typing import List

from graphtorch.core import Graph
from graphtorch.core.utils import split_node_keys


def connect_parallel(graph: Graph, nodes_per_hidden_layer: List):
    """
    Connects the graph to the next layer.

    Args:
        graph (Graph) :
        nodes_per_hidden_layer (list) :


    """
    #
    # Get node keys
    #
    node_keys_input, node_keys_hidden, node_keys_output = split_node_keys(
        graph.node_keys,
        nodes_per_hidden_layer,
    )
    #
    # Connect all input node to hidden node at layer 0
    #
    layer_idx_to = 0
    for node_key_input_from in node_keys_input:
        for node_key_hidden_to in node_keys_hidden[layer_idx_to]:
            graph[node_key_input_from, node_key_hidden_to] = True
    #
    # Connect all hidden nodes to next hidden node
    #
    for layer_idx_from, node_keys_hidden_from in zip(
        list(node_keys_hidden.keys())[:-1],
        list(node_keys_hidden.values())[:-1],
    ):
        layer_idx_to = layer_idx_from + 1
        node_keys_hidden_to = node_keys_hidden[layer_idx_to]
        for node_key_hidden_from in node_keys_hidden_from:
            for node_key_hidden_to in node_keys_hidden_to:
                graph[node_key_hidden_from, node_key_hidden_to] = True
    #
    # Connect last hidden node to output node
    #
    layer_idx_from = len(nodes_per_hidden_layer) - 1
    for node_key_output_to in node_keys_output:
        for node_key_hidden_from in node_keys_hidden[layer_idx_from]:
            graph[node_key_hidden_from, node_key_output_to] = True
    return graph
