from typing import Dict
from typing import Iterable
from typing import Union

import torch.nn as nn
from torch import Tensor

from graphtorch.graph import Graph


class GraphLayer(nn.Module):
    def __init__(self, graph: Graph):
        super().__init__()
        #
        # Infos defining network
        #
        self.node_keys = graph.node_keys
        self.node_dims = graph.node_dims
        self.adjacency_matrix = graph.adjacency_matrix
        self.feature_matrix = graph.feature_matrix
        #
        # Actual pytorch objects have parameters
        #
        self.wires = nn.ModuleDict()
        for node_idx_from, node_key_from in enumerate(self.node_keys):
            for node_idx_to, node_key_to in enumerate(self.node_keys[1:], start=1):
                if self.adjacency_matrix.loc[node_key_from, node_key_to] is True:
                    in_dim = self.node_dims[node_idx_from]
                    out_dim = self.node_dims[node_idx_to]
                    self.wires[
                        "%s_%s" % (node_key_from, node_key_to)
                    ] = self.feature_matrix["wire"].loc[node_key_from, node_key_to](
                        in_dim, out_dim,
                    )

    def forward(self, x: Tensor):
        #
        # Inputs from x (if needed, split)
        #
        inputs = {}
        node_keys_input = [x for x in self.node_keys if "I:" in x]
        if len(node_keys_input) == 1:
            inputs["I:0"] = x
        else:
            for node_idx, node_key in enumerate(node_keys_input):
                inputs[node_key] = x[:, node_idx].reshape(-1, 1)
        #
        # All other nodes from previous nodes
        #
        for node_key_to in self.node_keys[len(node_keys_input) :]:
            x_partial_sum = None
            for node_key_from in self.node_keys[0 : self.node_keys.index(node_key_to)]:
                if self.adjacency_matrix.loc[node_key_from, node_key_to] is True:
                    wire = self.wires["%s_%s" % (node_key_from, node_key_to)]
                    if x_partial_sum is None:
                        x_partial_sum = wire(inputs[node_key_from])
                    else:
                        x_partial_sum += wire(inputs[node_key_from])
            inputs[node_key_to] = x_partial_sum
        #
        # Outputs (if needed, split)
        #
        node_keys_output = [x for x in self.node_keys if "O:" in x]
        if len(node_keys_output) == 1:
            outputs = inputs["O:0"]
        else:
            for node_key in node_keys_output:
                outputs = torch.cat((outputs, inputs[node_key]), 1)
        return outputs
