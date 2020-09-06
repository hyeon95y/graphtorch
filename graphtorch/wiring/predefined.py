def input_to_all(connection, layer_sparsity=0):
    assert layer_sparsity <= 1 and layer_sparsity >= 0
    input_node_keys = connection.filter(regex="I:").columns.tolist()
    other_node_keys = list(set(connection.columns) - set(input_node_keys))

    for input_node_key in input_node_keys:
        for other_node_key in other_node_keys:
            connection.loc[input_node_key, other_node_key] = 1 - layer_sparsity

    return connection


def all_to_output(connection, layer_sparsity=0):
    assert layer_sparsity <= 1 and layer_sparsity >= 0
    output_node_keys = connection.filter(regex="O:").columns.tolist()
    other_node_keys = list(set(connection.columns) - set(output_node_keys))

    for output_node_key in output_node_keys:
        for other_node_key in other_node_keys:
            connection.loc[other_node_key, output_node_key] = 1 - layer_sparsity

    return connection


def all_to_all(connection, layer_sparsity=0, parallel=False, nodes_per_depth=None):
    assert layer_sparsity <= 1 and layer_sparsity >= 0
    if not parallel:
        node_keys = connection.columns.tolist()
        for node_key_from in node_keys:
            nodes_to_connect = node_keys[node_keys.index(node_key_from) + 1 :]
            for node_key_to in nodes_to_connect:
                connection.loc[node_key_from, node_key_to] = 1 - layer_sparsity
    else:
        assert nodes_per_depth is not None
        input_nodes = connection.filter(regex="I:").columns.tolist()
        hidden_nodes = connection.filter(regex="H:").columns.tolist()
        output_nodes = connection.filter(regex="O:").columns.tolist()
        all_nodes = connection.columns.tolist()
        #
        # Input nodes
        #
        for input_node in input_nodes:
            for node_to in all_nodes:
                connection.loc[input_node, node_to] = 1 - layer_sparsity
        #
        # Hidden nodes
        #
        for depth_from, nodes_from in zip(
            nodes_per_depth.keys(), nodes_per_depth.values()
        ):
            last_node_in_depth = nodes_from[-1]
            all_other_nodes = all_nodes[all_nodes.index(last_node_in_depth) + 1 :]
            for node_from in nodes_from:
                for node_to in all_other_nodes:
                    connection.loc[node_from, node_to] = 1 - layer_sparsity

    return connection
