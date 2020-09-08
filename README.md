<p align="center">
  <img width="1272" alt="image" src="https://www.jaronsanders.nl/wp-content/uploads/2020/02/Neural-network-with-Dropout-1020x642.png">
</p>

<p align="center">
  <code>graphtorch</code> is a <b>tool for easy-generating neural networks from adjacency matrix</b>
</p>

## 🗨️ Usage
- Easily build modules with pre-defined wiring


## ⚡️ Quickstart
```python

# Defined your specification of wiring
>> in_dim = 2
>> out_dim = 3
>> nodes_dim = [10, 20, 30, 50, 70]

# Define your 'connection' accepts [in_dim, out_dim]
>> import torch.nn as nn
>> def wire(in_dim, out_dim):
>>    layer = []
>>    layer += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
>>    return nn.Sequential(*layer)

# Create WiredLayer to your taste
>> from graphtorch.wiring.sequential import create_sequential_connections
>> connections_info = create_sequential_connections(
>>    node_dims=nodes_dim,
>>    in_dim=in_dim,
>>    out_dim=out_dim,
>>    split_input_layer=False,
>>    split_output_layer=False,
>> )
>> connections_info["connection"].loc["H:0", "H:2"] = 1
>> connections_info["connection"].loc["H:1", "H:4"] = 1
>> connections_info["connection"].loc["H:1", "O:0"] = 1

>>display(connections_info["connection"])
	I:0	H:0	H:1	H:2	H:3	H:4	O:0
FROM-TO							
I:0	NaN	1.0	NaN	NaN	NaN	NaN	NaN
H:0	NaN	NaN	1.0	1.0	NaN	NaN	NaN
H:1	NaN	NaN	NaN	1.0	NaN	1.0	1.0
H:2	NaN	NaN	NaN	NaN	1.0	NaN	NaN
H:3	NaN	NaN	NaN	NaN	NaN	1.0	NaN
H:4	NaN	NaN	NaN	NaN	NaN	NaN	1.0
O:0	NaN	NaN	NaN	NaN	NaN	NaN	NaN
>>print(connections_info["dimension"])
[2, 10, 20, 30, 50, 70, 3]

>> import graphtorch as gt
>> layer = gt.WiredLayer(
    connections_info, wire=wire, wire_to_output=nn.Linear, sanity_check=True
)
>> print(layer)
WiredLayer(
  (wires): ModuleDict(
    (I:0_H:0): Sequential(
      (0): Linear(in_features=2, out_features=10, bias=True)
      (1): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:0_H:1): Sequential(
      (0): Linear(in_features=10, out_features=20, bias=True)
      (1): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:0_H:2): Sequential(
      (0): Linear(in_features=10, out_features=30, bias=True)
      (1): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:1_H:2): Sequential(
      (0): Linear(in_features=20, out_features=30, bias=True)
      (1): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:1_H:4): Sequential(
      (0): Linear(in_features=20, out_features=70, bias=True)
      (1): BatchNorm1d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:1_O:0): Linear(in_features=20, out_features=3, bias=True)
    (H:2_H:3): Sequential(
      (0): Linear(in_features=30, out_features=50, bias=True)
      (1): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:3_H:4): Sequential(
      (0): Linear(in_features=50, out_features=70, bias=True)
      (1): BatchNorm1d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (H:4_O:0): Linear(in_features=70, out_features=3, bias=True)
  )
)
```

