def get_node_keys(
    in_dim: int,
    out_dim: int,
    hidden_node_dims: list,
    split_input_layer: bool = False,
    split_output_layer: bool = False,
    concat: bool = True,
):

    node_keys = []
    node_keys += ["I:%d" % x for x in range(in_dim)] if split_input_layer else ["I:0"]
    node_keys += ["H:%d" % x for x in range(len(hidden_node_dims))]
    node_keys += ["O:%d" % x for x in range(out_dim)] if split_output_layer else ["O:0"]

    if concat:
        return node_keys
    else:
        return split_node_keys(node_keys)


def split_node_keys(node_keys: list, nodes_per_hidden_layer: list = None):
    node_keys_input = [x for x in node_keys if "I:" in x]
    node_keys_output = [x for x in node_keys if "O:" in x]
    node_keys_hidden = [x for x in node_keys if "H:" in x]
    if nodes_per_hidden_layer is None:
        return node_keys_input, node_keys_hidden, node_keys_output

    else:
        node_keys_hidden_by_layer = {}
        node_idx_from, node_idx_to = 0, 0
        for layer_idx, num_nodes_per_hidden_layer in enumerate(nodes_per_hidden_layer):
            node_idx_to = node_idx_from + num_nodes_per_hidden_layer
            node_keys_hidden_by_layer[layer_idx] = node_keys_hidden[
                node_idx_from:node_idx_to
            ]
            node_idx_from += num_nodes_per_hidden_layer
        return node_keys_input, node_keys_hidden_by_layer, node_keys_output


def sort_node_keys(node_keys: list):
    node_keys_input = [x for x in node_keys if "I:" in x]
    node_keys_hidden = [x for x in node_keys if "H:" in x]
    node_keys_output = [x for x in node_keys if "O:" in x]
    node_keys_input.sort()
    node_keys_hidden.sort()
    node_keys_output.sort()
    node_keys_sorted = node_keys_input + node_keys_hidden + node_keys_output
    return node_keys_sorted


def get_node_dims(
    in_dim: int,
    out_dim: int,
    hidden_node_dims: list,
    split_input_layer: bool = False,
    split_output_layer: bool = False,
    concat: bool = True,
):

    node_dims_input = [1 for x in range(in_dim)] if split_input_layer else [in_dim]
    node_dims_hidden = hidden_node_dims
    node_dims_output = [1 for x in range(out_dim)] if split_output_layer else [out_dim]

    if concat:
        return node_dims_input + node_dims_hidden + node_dims_output
    else:
        return node_dims_input, node_dims_hidden, node_dims_output
