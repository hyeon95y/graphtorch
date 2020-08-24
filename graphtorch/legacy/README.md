# graphtorch
Package converts sparse graph matrix to PyTorch model

## Installation

```
pip install graphtorch
```

## [Examples](https://github.com/KU-BIG/graphtorch/blob/master/example.ipynb)

### Create sparse matrix with essential impormation
```python
from graphtorch import SpraseMatrix

mat1 = np.array([[0,2,0,0,2,0,0,0,0,0],
                [2,0,2,0,0,0,0,0,0,0],
                [0,2,0,2,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0],
                [0,0,0,0,0,0,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,3],
                [0,0,0,0,0,0,0,0,3,0]])  
in_dim = 5   
out_dim = 2  
mat_wann1 = SparseMatrix(mat1, in_dim, out_dim)   
```

### Create sparse torch model with SparseMatrix
```python
from graphtorch import SparseModel

activations = [None, None, nn.ReLU(), nn.Sigmoid()]  
constant_weight = 1 
model = SparseModel(mat_wann1, activations, constant_weight)

numpy_input = np.array([[1,2,3,4,5],  
                        [6,7,8,9,10],  
                        [11,12,13,14,15]])      

numpy_input = torch.from_numpy(numpy_input).float()  
output, nodes = model(numpy_input)  
```


### output


```python
tensor([[1.0000, 1.0000],
        [1.0000, 1.0000],
        [1.0000, 1.0000]], grad_fn=<CatBackward>)
```

### nodes

```python
{'hidden_0': tensor([[ 7.],
         [17.],
         [27.]], grad_fn=<AddBackward0>), 'hidden_1': tensor([[ 4.],
         [14.],
         [24.]], grad_fn=<AddBackward0>), 'hidden_2': tensor([[ 6.],
         [16.],
         [26.]], grad_fn=<AddBackward0>), 'hidden_3': tensor([[11.],
         [31.],
         [51.]], grad_fn=<AddBackward0>), 'hidden_4': tensor([[10.],
         [30.],
         [50.]], grad_fn=<AddBackward0>), 'output_0': tensor([[1.0000],
         [1.0000],
         [1.0000]], grad_fn=<SigmoidBackward>), 'output_1': tensor([[1.0000],
         [1.0000],
         [1.0000]], grad_fn=<SigmoidBackward>)}
```

## Maintainer
- Sehee Lee (lsh9382@gmail.com / lsh9382@korea.ac.kr)     
- Hyeonwoo Yoo (hyeon95y@gmail.com)    

---- 

# Draft

## Still working on code refactoring and documentation

### References for documentation
- [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning)
    - [raw](https://raw.githubusercontent.com/williamFalcon/pytorch-lightning/master/README.md)
- [sentencepiece](https://github.com/google/sentencepiece)
    - [raw](https://raw.githubusercontent.com/google/sentencepiece/master/README.md)

