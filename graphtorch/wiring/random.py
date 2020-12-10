import os
import random
from typing import Dict
from typing import Iterable
from typing import Union

import numpy as np
import torch

from graphtorch.graph import Graph
from graphtorch.graph.utils import get_node_keys
from graphtorch.graph.utils import split_node_keys
from graphtorch.wiring import connect_parallel
from graphtorch.wiring.utils import get_all_possible_connections
from graphtorch.wiring.utils import get_network_sparsity
from graphtorch.wiring.utils import in_hidden_out_dim_from_graph
from graphtorch.wiring.utils import in_same_depth
from graphtorch.wiring.utils import maximum_num_of_connections


def connect_randomly(graph: Graph, seed: int = 0, network_sparsity: float = 0.4):
    #
    # Fix all seeds
    #
    fix_everything(seed)
    #
    # Define some random things
    #
    nodes_per_hidden_layer = distribute_nodes_randomly(graph.hidden_node_dims)
    #
    # Initial connection
    #
    graph = connect_parallel(graph, nodes_per_hidden_layer)
    #
    # Randomly add connections until converges to predefined netowrk sparsity
    #
    graph = add_connections_until_target_network_sparsity(
        graph,
        network_sparsity,
        nodes_per_hidden_layer,
    )

    return graph


def fix_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def distribute_nodes_randomly(hidden_node_dims: list):
    num_layers = random.randint(1, len(hidden_node_dims))

    nodes_per_hidden_layer = []
    num_nodes_used = 0

    for idx_layer in range(num_layers - 1):
        num_nodes_min = 1
        num_nodes_max = (
            len(hidden_node_dims) - (num_layers - idx_layer) - num_nodes_used + 1
        )
        if num_nodes_max == 1:
            num_nodes_in_current_layer = 1
        else:
            num_nodes_in_current_layer = np.random.randint(num_nodes_min, num_nodes_max)

        nodes_per_hidden_layer.append(num_nodes_in_current_layer)
        num_nodes_used += num_nodes_in_current_layer

    nodes_per_hidden_layer.append(len(hidden_node_dims) - num_nodes_used)

    return nodes_per_hidden_layer


def add_connections_until_target_network_sparsity(
    graph: Graph,
    network_sparsity: float,
    nodes_per_hidden_layer: list,
):
    current_network_sparsity = get_network_sparsity(graph, nodes_per_hidden_layer)

    (
        all_possible_connections,
        all_possible_connections_key,
    ) = get_all_possible_connections(graph, nodes_per_hidden_layer)

    while current_network_sparsity > network_sparsity:

        new_connection_key = np.random.choice(
            range(0, all_possible_connections_key),
            replace=False,
        )
        new_connection = all_possible_connections[new_connection_key]
        node_from, node_to = new_connection[0], new_connection[1]
        graph[node_from, node_to] = True

        current_network_sparsity = get_network_sparsity(graph, nodes_per_hidden_layer)

    return graph
