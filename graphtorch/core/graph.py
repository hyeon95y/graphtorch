from typing import Dict
from typing import Iterable
from typing import Union

import pandas as pd
import torch.nn as nn

from graphtorch.graph.utils import get_node_dims
from graphtorch.graph.utils import get_node_keys


class Graph:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_node_dims: list,
        split_input_layer: bool = False,
        split_output_layer: bool = False,
        global_wire_sparsity: float = 1,
        global_wire: object = None,
        global_wire_to_output: object = None,
    ):
        #
        # Config
        #
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_node_dims = hidden_node_dims
        self.global_wire_sparsity = global_wire_sparsity
        self.global_wire = global_wire if global_wire is not None else default_wire
        self.global_wire_to_output = (
            global_wire_to_output
            if global_wire_to_output is not None
            else default_wire_to_output
        )
        #
        # Keys, dims
        #
        self.node_keys = get_node_keys(
            in_dim,
            out_dim,
            hidden_node_dims,
            split_input_layer,
            split_output_layer,
        )
        self.node_dims = get_node_dims(
            in_dim,
            out_dim,
            hidden_node_dims,
            split_input_layer,
            split_output_layer,
        )
        #
        # Matrixes defining network
        #
        self.adjacency_matrix = empty_matrix(
            in_dim,
            out_dim,
            hidden_node_dims,
            split_input_layer,
            split_output_layer,
        )
        self.feature_keys = ["sparsity", "wire", "dimension"]
        self.feature_matrix = {}
        for feature_key in self.feature_keys:
            self.feature_matrix[feature_key] = empty_matrix(
                in_dim,
                out_dim,
                hidden_node_dims,
                split_input_layer,
                split_output_layer,
            )
            if feature_key == "dimension":
                for node_idx, node_key in enumerate(self.node_keys):
                    self.feature_matrix[feature_key].loc[
                        node_key, node_key
                    ] = self.node_dims[node_idx]

    def __getitem__(self, index: tuple):
        if index[0] in self.node_keys and index[1] in self.node_keys:
            item = {
                "adjacency_matrix": self.adjacency_matrix.loc[index],
                "feature_matrix": {
                    "sparsity": self.feature_matrix["sparsity"].loc[index],
                    "wire": self.feature_matrix["wire"].loc[index],
                    "dimension": self.feature_matrix["dimension"].loc[index],
                },
            }
        else:
            raise IndexError("%s, %s not in matrix" % (index[0], index[1]))
        return item

    def __setitem__(self, index: tuple, value: bool):
        if index[0] in self.node_keys and index[1] in self.node_keys:
            self.adjacency_matrix.loc[index] = value
            self.feature_matrix["sparsity"].loc[index] = self.global_wire_sparsity
            if [x for x in index if "O:" in x]:
                self.feature_matrix["wire"].loc[index] = self.global_wire_to_output
            else:
                self.feature_matrix["wire"].loc[index] = self.global_wire
        else:
            raise IndexError("%s, %s not in matrix" % (index[0], index[1]))
        return


def empty_matrix(
    in_dim: int,
    out_dim: int,
    hidden_node_dims: list,
    split_input_layer: bool = False,
    split_output_layer: bool = False,
):

    nodes = get_node_keys(
        in_dim,
        out_dim,
        hidden_node_dims,
        split_input_layer,
        split_output_layer,
    )

    matrix = pd.DataFrame(columns=nodes, index=nodes).rename_axis("FROM\TO")
    return matrix


def default_wire(in_dim: int, out_dim: int):
    layer = []
    layer += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
    return nn.Sequential(*layer)


def default_wire_to_output(in_dim: int, out_dim: int):
    layer = []
    layer += [nn.Linear(in_dim, out_dim)]
    return nn.Sequential(*layer)
