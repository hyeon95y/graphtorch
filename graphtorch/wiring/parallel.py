import copy

import numpy as np
import pandas as pd
from graphtorch.wiring.utils import create_empty_connection_dataframe
from graphtorch.wiring.utils import split_nodes_by_depth


def connect_parallel_connections(
    empty_connection,
    in_dim,
    out_dim,
    num_nodes_in_depth,
    split_input_layer,
    split_ouput_layer,
    layer_sparsity,
):
    #
    # Input sanity check
    #
    assert empty_connection.shape[0] == empty_connection.shape[1]
    #
    # Create parallel connection object
    #
    parallel_connection = empty_connection
    #
    # Get input, hiddne, output nodes
    #
    input_nodes = parallel_connection.filter(regex="I:").columns.tolist()
    hidden_nodes = parallel_connection.filter(regex="H:").columns.tolist()
    output_nodes = parallel_connection.filter(regex="O:").columns.tolist()
    all_nodes = parallel_connection.columns.tolist()
    #
    # Assign actual name of nodes by given num_nodes_in_depth
    #
    nodes_per_depth = split_nodes_by_depth(hidden_nodes, num_nodes_in_depth)
    #
    # Create connection
    #
    num_total_connections = 0
    # Input nodes
    for input_node in input_nodes:
        for node_to in nodes_per_depth[0]:
            parallel_connection.loc[input_node, node_to] = 1 - layer_sparsity
            num_total_connections += 1
    # Hidden nodes
    for depth_from, nodes_from in zip(nodes_per_depth.keys(), nodes_per_depth.values()):
        if depth_from != len(num_nodes_in_depth) - 1:
            depth_to = depth_from + 1
            nodes_to = nodes_per_depth[depth_to]
            print(nodes_from, nodes_to)
            for node_from in nodes_from:
                for node_to in nodes_to:
                    parallel_connection.loc[node_from, node_to] = 1 - layer_sparsity
                    num_total_connections += 1
    # Output nodes
    for output_node in output_nodes:
        last_depth = len(num_nodes_in_depth) - 1
        for node_from in nodes_per_depth[last_depth]:
            parallel_connection.loc[node_from, output_node] = 1 - layer_sparsity
            num_total_connections += 1

    return parallel_connection, nodes_per_depth, num_total_connections


def create_parallel_connections(
    node_dims,
    in_dim,
    out_dim,
    num_nodes_in_depth,
    split_input_layer=False,
    split_output_layer=False,
    layer_sparsity=0,
):
    empty_connection = create_empty_connection_dataframe(
        node_dims, in_dim, out_dim, split_input_layer, split_output_layer,
    )
    (
        parallel_connection,
        nodes_per_depth,
        num_total_connections,
    ) = connect_parallel_connections(
        empty_connection,
        in_dim,
        out_dim,
        num_nodes_in_depth,
        split_input_layer,
        split_output_layer,
        layer_sparsity,
    )

    dimension = []
    dimension += [1 for x in range(in_dim)] if split_input_layer else [in_dim]
    dimension += node_dims
    dimension += [1 for x in range(out_dim)] if split_output_layer else [out_dim]

    connections = {
        "connection": parallel_connection,
        "dimension": dimension,
        "nodes_per_depth": nodes_per_depth,
        "num_total_connections": num_total_connections,
    }

    return connections
