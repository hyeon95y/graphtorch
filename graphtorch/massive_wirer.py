import torch
import torch.nn as nn

from graphtorch.sanity_check import wires_sanity_check


class WiredLayer(nn.Module):
    def __init__(
        self,
        connections,
        wire=None,
        wire_to_output=None,
        sanity_check=True,
        sequential=False,
    ):
        super().__init__()

        # Tensor info
        self.sequential = (
            sequential  # if seqeutnial is true, feature comes at dimension 2, else 1
        )

        # Wire infos
        self.connection = connections["connection"]
        self.node_keys = self.connection.columns.tolist()
        self.dimension = connections["dimension"]
        self.wire = wire
        self.wire_to_output = wire_to_output
        if "wires" in connections:
            self.wires_assigned = connections["wires"]
            self.wires_dict = connections["wires_dict"]
        # Nodes
        self.input_node_keys = self.connection.filter(regex="I:").columns.tolist()
        self.input_node_dims = self.dimension[: len(self.input_node_keys)]
        self.output_node_keys = self.connection.filter(regex="O:").columns.tolist()
        self.output_node_dims = self.dimension[-len(self.output_node_keys) :]
        self.hidden_node_keys = self.connection.filter(regex="H:").columns.tolist()
        self.hidden_node_dims = self.dimension[
            len(self.input_node_keys) : -len(self.output_node_keys)
        ]
        # Wires
        self.wires = nn.ModuleDict()
        for idx_from, node_key_from in enumerate(self.node_keys):
            for idx_to, node_key_to in enumerate(self.node_keys[1:], start=1):
                if self.connection.loc[node_key_from, node_key_to] > 0:
                    in_dim = self.dimension[idx_from]
                    out_dim = self.dimension[idx_to]
                    if wire is not None:
                        if wire_to_output is not None and "O:" in node_key_to:
                            self.wires[
                                "%s_%s" % (node_key_from, node_key_to)
                            ] = self.wire_to_output(in_dim, out_dim)
                        else:
                            self.wires[
                                "%s_%s" % (node_key_from, node_key_to)
                            ] = self.wire(in_dim, out_dim)
                    else:
                        wire_to_use = self.wires_assigned.loc[
                            node_key_from, node_key_to
                        ]
                        self.wires[
                            "%s_%s" % (node_key_from, node_key_to)
                        ] = self.wires_dict[wire_to_use](in_dim, out_dim)

        # Sanity check
        for check_result in wires_sanity_check(self.connection, self.dimension):
            print(check_result[0], check_result[1])

    def forward(self, x):
        inputs = {}
        if len(self.input_node_keys) == 1:
            inputs["I:0"] = x
        else:
            for input_node_idx, input_node_key in enumerate(self.input_node_keys):
                inputs[input_node_key] = x[:, input_node_idx].reshape(-1, 1)
        for node_key_to in self.node_keys[len(self.input_node_keys) :]:
            x_sum = None
            for node_key_from in self.node_keys[0 : self.node_keys.index(node_key_to)]:
                if self.connection.loc[node_key_from, node_key_to] > 0:
                    wire = self.wires["%s_%s" % (node_key_from, node_key_to)]
                    if x_sum is None:
                        x_sum = wire(inputs[node_key_from])
                    else:
                        x_sum += wire(inputs[node_key_from])
            inputs[node_key_to] = x_sum

        outputs = inputs["O:0"]
        if len(self.output_node_keys) != 1:
            for output_node_key in self.output_node_keys:
                if output_node_key != "O:0":
                    if self.sequential:
                        outputs = torch.cat((outputs, inputs[output_node_key]), 2)
                    else:
                        outputs = torch.cat((outputs, inputs[output_node_key]), 1)

        return outputs
