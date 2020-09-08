<p align="center">
  <img width="1272" alt="image" src="https://user-images.githubusercontent.com/24773116/75319337-d1c39b80-58af-11ea-83f1-6c0d2a0bbad1.png">
  <img width="80" alt="image" src="https://user-images.githubusercontent.com/24773116/75319298-bc4e7180-58af-11ea-8df2-eac383cdad73.png">
</p>

<p align="center">
  <code>graphtorch</code> is a <b>tool for easy-generating neural networks with skip connections</b>
</p>

## 🗨️ Usage
- Easily build modules with pre-defined wiring


## ⚡️ Quickstart
```python
>> from nn_wirer.NN import LayerwiseWiredNN
>> from nn_wirer.wires import create_initial_wires

>> hidden_layers_info = [10, 20, 30]
>> num_hidden_layers = len(hidden_layers_info)
>> wires_info = create_initial_wires(num_hidden_layers)
>> wires_info.loc[0, 3] = True
>> wires_info.loc[1, 4] = True
>> wires_info
	0	1	2	3	4
FROM-TO					
0	NaN	True	False	True	False
1	NaN	NaN	True	False	True
2	NaN	NaN	NaN	True	False
3	NaN	NaN	NaN	NaN	True
4	NaN	NaN	NaN	NaN	NaN

>> model = LayerwiseWiredNN(
        input_shape=20,
        output_shape=1,
        hidden_layers_info=hidden_layers_info,
        wires_info=wires_info,
    )
>> model
LayerwiseWiredNN(
  (layers): ModuleDict(
    (layer_0_1): Sequential(
      (0): Linear(in_features=20, out_features=10, bias=True)
      (1): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (layer_0_3): Sequential(
      (0): Linear(in_features=20, out_features=30, bias=True)
      (1): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (layer_1_2): Sequential(
      (0): Linear(in_features=10, out_features=20, bias=True)
      (1): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (layer_1_4): Sequential(
      (0): Linear(in_features=10, out_features=1, bias=True)
    )
    (layer_2_3): Sequential(
      (0): Linear(in_features=20, out_features=30, bias=True)
      (1): BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (layer_3_4): Sequential(
      (0): Linear(in_features=30, out_features=1, bias=True)
    )
  )
)

```
- hidden_layers_info 
  - number of nodes in each hidden layer
- wires_info
  - define where from/to connect wire



