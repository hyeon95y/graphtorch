from typing import List

from graphtorch.core import Graph
from graphtorch.core.utils import get_node_keys


def in_hidden_out_dim_from_graph(graph: Graph):
    """
    Extract the in - hidden dimension from the graph.

    Args:
        graph (Graph) :


    """
    node_keys = graph.adjacency_matrix.columns
    split_input_layer = True if len([x for x in node_keys if "I:" in x]) > 1 else False
    split_output_layer = True if len([x for x in node_keys if "O:" in x]) > 1 else False
    if split_input_layer is True:
        in_dim = len([x for x in node_keys if "I:" in x])
    else:
        in_dim = graph.feature_matrix["dimension"].loc["I:0", "I:0"]
    if split_output_layer is True:
        out_dim = len([x for x in node_keys if "O:" in x])
    else:
        out_dim = graph.feature_matrix["dimension"].loc["O:0", "O:0"]
    hidden_node_keys = [x for x in node_keys if "H:" in x]
    hidden_node_dims = [
        graph.feature_matrix["dimension"].loc[x, x]
        for x in hidden_node_keys
    ]

    return in_dim, hidden_node_dims, out_dim, split_input_layer, split_output_layer


def in_same_depth(node1: str, node2: str, nodes_per_hidden_layer: List):
    """
    Check if two nodes are in same depth.

    Args:
        node1 (str) :
        node2 (str) :
        nodes_per_hidden_layer (list) :


    """
    if "I:" in node1 and "I:" in node2:
        return True
    elif "O:" in node1 and "O:" in node2:
        return True
    elif "I:" in node1 and "I:" not in node2:
        return False
    elif "O:" in node1 and "O:" not in node2:
        return False
    elif "I:" not in node1 and "I:" in node2:
        return False
    elif "O:" not in node1 and "O:" in node2:
        return False
    else:
        depth_node1 = get_depth(node1, nodes_per_hidden_layer)
        depth_node2 = get_depth(node2, nodes_per_hidden_layer)
        if depth_node1 == depth_node2:
            return True
        else:
            return False


def get_depth(node: str, nodes_per_hidden_layer: List):
    """
    find the depth of a given node

    Args:
        node (str) :
        nodes_per_hidden_layer (list) :


    """
    num_nodes_before_me = 0

    for layer_idx in range(len(nodes_per_hidden_layer)):
        num_nodes_before_me += nodes_per_hidden_layer[layer_idx]
        if int(node.replace("H:", "")) >= num_nodes_before_me:
            return layer_idx


def get_all_possible_connections(graph: Graph, nodes_per_hidden_layer: List):
    """
    Get all possible connections for a graph.

    Args:
        graph (Graph) :
        nodes_per_hidden_layer (list) :


    """
    (
        in_dim,
        hidden_node_dims,
        out_dim,
        split_input_layer,
        split_output_layer,
    ) = in_hidden_out_dim_from_graph(graph)
    node_keys_all = get_node_keys(
        in_dim,
        out_dim,
        hidden_node_dims,
        split_input_layer,
        split_output_layer,
    )

    all_possible_connections = {}
    all_possible_connections_key = 0

    for node_key_from in node_keys_all:
        all_other_nodes_to = node_keys_all[node_keys_all.index(node_key_from) + 1:]
        if len(all_other_nodes_to) != 0:
            for node_key_to in all_other_nodes_to:
                if not in_same_depth(
                    node1=node_key_from,
                    node2=node_key_to,
                    nodes_per_hidden_layer=nodes_per_hidden_layer,
                ):
                    if graph.adjacency_matrix.isna().loc[node_key_from, node_key_to]:
                        all_possible_connections[all_possible_connections_key] = [
                            node_key_from,
                            node_key_to,
                        ]
                        all_possible_connections_key += 1

    return all_possible_connections, all_possible_connections_key


def get_network_sparsity(graph: Graph, nodes_per_hidden_layer: List):
    """
    Returns the network sparsity.

    Args:
        graph (Graph) :
        nodes_per_hidden_layer (list) :


    """
    num_current_connections = graph.adjacency_matrix.sum().sum()
    num_max_connections = maximum_num_of_connections(graph, nodes_per_hidden_layer)

    current_network_sparsity = 1 - (num_current_connections / num_max_connections)

    return current_network_sparsity


def maximum_num_of_connections(graph: Graph, nodes_per_hidden_layer: List):
    """
    Maximum number of connections in the network.

    Args:
        graph (Graph) :
        nodes_per_hidden_layer (list) :


    """
    (
        in_dim,
        _,
        out_dim,
        split_input_layer,
        split_output_layer,
    ) = in_hidden_out_dim_from_graph(graph)

    max_connections = 0
    nodes_per_dim = []
    if split_input_layer:
        nodes_per_dim += [in_dim]
    else:
        nodes_per_dim += [1]
    nodes_per_dim += nodes_per_hidden_layer
    if split_output_layer:
        nodes_per_dim += [out_dim]
    else:
        nodes_per_dim += [1]
    for idx_from, num_nodes_from in enumerate(nodes_per_dim):

        node_keys_all_after_me = 0
        for idx_to, num_nodes_to in enumerate(nodes_per_dim[idx_from + 1:]):
            node_keys_all_after_me += num_nodes_to
        max_connections += num_nodes_from * node_keys_all_after_me

    return max_connections
