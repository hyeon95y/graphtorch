import copy

import numpy as np
import pandas as pd



def create_empty_connection_dataframe(
    node_dims, in_dim, out_dim, split_input_layer, split_output_layer,
):
    # Size of connections
    num_hidden_layers = len(node_dims)
    num_nodes = num_hidden_layers
    node_ids = []

    # Split condition check
    if split_input_layer:
        num_nodes += in_dim
        node_ids += ["I:%d" % x for x in range(in_dim)]
    else:
        num_nodes += 1
        node_ids += ["I:0"]

    node_ids += ["H:%d" % x for x in range(num_hidden_layers)]

    if split_output_layer:
        num_nodes += out_dim
        node_ids += ["O:%d" % x for x in range(out_dim)]

    else:
        num_nodes += 1
        node_ids += ["O:0"]

    # Create empty connection dataframe
    empty_connection = np.empty((num_nodes, num_nodes), dtype=bool)
    empty_connection = pd.DataFrame(empty_connection)
    empty_connection.columns = node_ids
    empty_connection.index = node_ids
    empty_connection = empty_connection.rename_axis("FROM-TO")
    empty_connection.loc[:, :] = np.NaN

    return empty_connection


def create_wire_dataframe(connections, wires_dict):
    wires = copy.deepcopy(connections["connection"])
    wires[wires.notnull()] = next(iter(wires_dict))

    connections["wires"] = wires
    connections["wires_dict"] = wires_dict
    return connections


def connection_to_list(connection):
    connection_list = []
    all_nodes = connection.columns.tolist()
    for idx_row, node_row in enumerate(all_nodes):
        for idx_col, node_col in enumerate(all_nodes[idx_row:]):
            if not connection.isna().loc[node_row, node_col]:
                connection_list.append([node_row, node_col])
    return connection_list


def split_nodes_by_depth(hidden_nodes, num_nodes_in_depth):
    idx_from, idx_to = 0, 0
    nodes_per_depth = {}
    for idx_depth, num_node_per_depth in enumerate(num_nodes_in_depth):
        idx_to = idx_from + num_node_per_depth
        nodes_per_depth[idx_depth] = hidden_nodes[idx_from:idx_to]
        idx_from += num_node_per_depth
    return nodes_per_depth


def sort_nodes(nodes):
    input_nodes = [x for x in nodes if "I:" in x]
    hidden_nodes = [x for x in nodes if "H:" in x]
    output_nodes = [x for x in nodes if "O:" in x]
    input_nodes.sort()
    hidden_nodes.sort()
    output_nodes.sort()
    nodes_sorted = input_nodes + hidden_nodes + output_nodes
    return nodes_sorted

def maximum_num_of_connections(
    in_dim,
    out_dim,
    split_input_layer,
    split_output_layer,
    parallel=False,
    num_nodes_in_depth=None,
):

    if parallel:
        max_connections = 0
        nodes_per_dim = []
        if split_input_layer :
            nodes_per_dim += [in_dim]
        else : 
            nodes_per_dim += [1]
        nodes_per_dim += num_nodes_in_depth
        if split_output_layer : 
            nodes_per_dim += [out_dim]
        else : 
            nodes_per_dim += [1]
        for idx_from, num_nodes_from in enumerate(nodes_per_dim) :

            all_nodes_after_me = 0
            for idx_to, num_nodes_to in enumerate(nodes_per_dim[idx_from+1:]) :
                all_nodes_after_me += num_nodes_to
            max_connections += num_nodes_from * all_nodes_after_me
    else:
        raise NotImplementedError()
    

    return max_connections


def in_same_depth(node1, node2, nodes_per_depth):
    if "I:" in node1 and "I:" in node2:
        return True
    elif "O:" in node1 and "O:" in node2:
        return True
    else:
        node1_idx_depth = None
        for idx_depth, nodes in zip(nodes_per_depth.keys(), nodes_per_depth.values()):
            if node1 in nodes and node2 in nodes:
                return True
    return False

