import copy

import numpy as np
import pandas as pd
from graphtorch.wiring.utils import create_empty_connection_dataframe
from graphtorch.wiring.parallel import connect_parallel_connections
from graphtorch.wiring import all_to_all
from graphtorch.wiring.utils import split_nodes_by_depth
from graphtorch.wiring.utils import sort_nodes
from graphtorch.wiring.utils import maximum_num_of_connections
from graphtorch.wiring.utils import in_same_depth

def connect_random_connections(
    empty_connection,
    in_dim,
    out_dim,
    split_input_layer,
    split_output_layer,
    layer_sparsity,
    network_sparsity,
    seed,
    max_depth
):
    assert layer_sparsity <= 1 and layer_sparsity >= 0
    #
    # Input sanity check
    #
    assert empty_connection.shape[0] == empty_connection.shape[1]
    #
    # Create parallel connection object
    #
    random_connection = empty_connection
    #
    # Sample num of depth
    #
    np.random.seed(seed)
    num_nodes = len(random_connection.filter(regex="H:").columns.tolist())
    if max_depth is not None : 
        num_depths = np.random.randint(1, min(max_depth, num_nodes))
    else : 
        num_depths = np.random.randint(1, num_nodes)
    #
    # Assign node for each depth
    #
    hidden_nodes = empty_connection.filter(regex="H:").columns.tolist()
    all_nodes = empty_connection.columns.tolist()
    num_nodes_used = 0
    num_nodes_in_depth = []
    # Assgin num of nodes per depth
    for idx_depth in range(num_depths - 1):
        num_nodes_min = 1
        num_nodes_max = num_nodes - num_nodes_used - (num_depths - (idx_depth))
        num_nodes_max = max(num_nodes_min, num_nodes_max)
        if num_nodes_min == num_nodes_max : 
            num_nodes_to_use = 1
        else : 
            num_nodes_to_use = np.random.randint(num_nodes_min, num_nodes_max)
        num_nodes_used += num_nodes_to_use
        num_nodes_in_depth.append(num_nodes_to_use)
    num_nodes_in_depth.append(num_nodes - num_nodes_used)
    #
    # Assign actual name of nodes by given num_nodes_in_depth
    #
    nodes_per_depth = split_nodes_by_depth(hidden_nodes, num_nodes_in_depth)
    #
    # Generate essential wires
    #
    num_total_connections = 0
    # Input layer
    input_nodes = random_connection.filter(regex="I:").columns.tolist()
    all_other_nodes = list(set(all_nodes) - set(input_nodes))
    all_other_nodes = sort_nodes(all_other_nodes)
    for input_node in input_nodes:
        node_to = np.random.choice(all_other_nodes)
        if random_connection.isna().loc[input_node, node_to]:
            random_connection.loc[input_node, node_to] = 1 - layer_sparsity
            num_total_connections += 1
    # Output layer
    output_nodes = random_connection.filter(regex="O:").columns.tolist()
    all_other_nodes = list(set(all_nodes) - set(output_nodes))
    all_other_nodes = sort_nodes(all_other_nodes)
    for output_node in output_nodes:
        node_from = np.random.choice(all_other_nodes)
        if random_connection.isna().loc[node_from, output_node]:
            random_connection.loc[node_from, output_node] = 1 - layer_sparsity
            num_total_connections += 1
    # Hidden layers
    for idx_depth, nodes_in_depth in zip(
        nodes_per_depth.keys(), nodes_per_depth.values()
    ):
        all_nodes_before = all_nodes[: all_nodes.index(nodes_in_depth[0])]
        all_nodes_after = all_nodes[all_nodes.index(nodes_in_depth[-1]) + 1 :]
        for node in nodes_in_depth:
            # Left side
            node_from = np.random.choice(all_nodes_before)
            if random_connection.isna().loc[node_from, node]:
                random_connection.loc[node_from, node] = 1 - layer_sparsity
                num_total_connections += 1
            # Right side
            node_to = np.random.choice(all_nodes_after)
            if random_connection.isna().loc[node, node_to]:
                random_connection.loc[node, node_to] = 1 - layer_sparsity
                num_total_connections += 1
    #
    # Generate additional wires
    #
    num_maximum_connections = maximum_num_of_connections(
        in_dim,
        out_dim,
        split_input_layer,
        split_output_layer,
        parallel=True,
        num_nodes_in_depth=num_nodes_in_depth,
    )
    current_network_sparsity = 1 - (num_total_connections / num_maximum_connections)
    #
    # Get candidates of possible connections
    #
    all_possible_connections = {}
    all_possible_connections_key = 0
    for node_from in all_nodes : 
        all_other_nodes_to = all_nodes[all_nodes.index(node_from) + 1 :]
        if len(all_other_nodes_to) != 0 :
            for node_to in all_other_nodes_to :
                if not in_same_depth(node1=node_from, node2=node_to, nodes_per_depth=nodes_per_depth) :
                    if random_connection.isna().loc[node_from, node_to] : 
                        all_possible_connections[all_possible_connections_key] = [node_from, node_to]
                        all_possible_connections_key += 1
    #
    # Connect until sparsity reaches desired level
    #
    while current_network_sparsity > network_sparsity : 
        new_connection_key = np.random.choice(range(0, all_possible_connections_key), replace=False)
        new_connection = all_possible_connections[new_connection_key]
        node_from, node_to = new_connection[0], new_connection[1]
        random_connection.loc[node_from, node_to] = 1 - layer_sparsity
        num_total_connections += 1
        current_network_sparsity = 1 - (num_total_connections/num_maximum_connections)

    return random_connection, nodes_per_depth, num_total_connections


def create_random_connections(
    node_dims,
    in_dim,
    out_dim,
    split_input_layer=False,
    split_output_layer=False,
    layer_sparsity=0,
    network_sparsity=0,
    seed=0,
    max_depth=None
):
    num_nodes = len(node_dims)
    empty_connection = create_empty_connection_dataframe(
        node_dims, in_dim, out_dim, split_input_layer, split_output_layer,
    )
    (
        random_connection,
        nodes_per_depth,
        num_total_connections,
    ) = connect_random_connections(
        empty_connection,
        in_dim,
        out_dim,
        split_input_layer,
        split_output_layer,
        layer_sparsity,
        network_sparsity,
        seed,
        max_depth
    )

    dimension = []
    dimension += [1 for x in range(in_dim)] if split_input_layer else [in_dim]
    dimension += node_dims
    dimension += [1 for x in range(out_dim)] if split_output_layer else [out_dim]

    connections = {
        "connection": random_connection,
        "dimension": dimension,
        "nodes_per_depth": nodes_per_depth,
        "num_total_connections": num_total_connections,
    }

    return connections

