from typing import Dict
from typing import Iterable
from typing import Union

from graphtorch.graph import Graph
from graphtorch.graph.utils import split_node_keys


def connect_sequential(graph: Graph):
    #
    # Get node keys
    #
    node_keys_input, node_keys_hidden, node_keys_output = split_node_keys(
        graph.node_keys,
    )
    #
    # Connect all input nodes to hidden node at layer 0
    #
    node_key_hidden_to = node_keys_hidden[0]
    for node_key_input_from in node_keys_input:
        graph[node_key_input_from, node_key_hidden_to] = True
    #
    # Connect all hidden nodes to next hidden node
    #
    for node_key_hidden_from, node_key_hidden_to in zip(
        node_keys_hidden,
        node_keys_hidden[1:],
    ):
        graph[node_key_hidden_from, node_key_hidden_to] = True
    #
    # Connect last hidden node to output node
    #
    node_key_hidden_from = node_keys_hidden[-1]
    for node_key_output_to in node_keys_output:
        graph[node_key_hidden_from, node_key_output_to] = True
    return graph
