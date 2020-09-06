import copy

import numpy as np
import pandas as pd
from graphtorch.wiring.utils import create_empty_connection_dataframe


def connect_sequential_connections(
    empty_connection,
    in_dim,
    out_dim,
    split_input_layer,
    split_output_layer,
    layer_sparsity,
):
    #
    # Input sanity check
    #
    assert empty_connection.shape[0] == empty_connection.shape[1]
    #
    # Create sequential connection object
    #
    sequential_connection = empty_connection
    #
    # Get input, hiddne, output nodes
    #
    input_nodes = sequential_connection.filter(regex="I:").columns.tolist()
    hidden_nodes = sequential_connection.filter(regex="H:").columns.tolist()
    output_nodes = sequential_connection.filter(regex="O:").columns.tolist()
    #
    # Create connection
    #
    # Input nodes
    num_total_connections = 0
    for input_node in input_nodes:
        hidden_node = hidden_nodes[0]
        sequential_connection.loc[input_node, hidden_node] = 1 - layer_sparsity
        num_total_connections += 1
    # Hideen nodes
    for idx_hidden, hidden_node in enumerate(hidden_nodes[:-1]):
        hidden_node_from = hidden_node
        hidden_note_to = hidden_nodes[idx_hidden + 1]
        sequential_connection.loc[hidden_node_from, hidden_note_to] = 1 - layer_sparsity
        num_total_connections += 1
    # Output nodes
    for output_node in output_nodes:
        hidden_node = hidden_nodes[-1]
        sequential_connection.loc[hidden_node, output_node] = 1 - layer_sparsity
        num_total_connections += 1

    return sequential_connection, num_total_connections


def create_sequential_connections(
    node_dims,
    in_dim,
    out_dim,
    split_input_layer=False,
    split_output_layer=False,
    layer_sparsity=0,
):
    empty_connection = create_empty_connection_dataframe(
        node_dims, in_dim, out_dim, split_input_layer, split_output_layer,
    )
    sequential_connection, num_total_connections = connect_sequential_connections(
        empty_connection,
        in_dim,
        out_dim,
        split_input_layer,
        split_output_layer,
        layer_sparsity,
    )

    dimension = []
    dimension += [1 for x in range(in_dim)] if split_input_layer else [in_dim]
    dimension += node_dims
    dimension += [1 for x in range(out_dim)] if split_output_layer else [out_dim]

    connections = {
        "connection": sequential_connection,
        "dimension": dimension,
        "num_total_connections": num_total_connections,
    }

    return connections
