#import modules needed  
import numpy as np
import torch 
import torch.nn as nn

#class for matrix information 
class SparseMatrix():
    
    def __init__(self, mat, in_dim, out_dim):
        # get when initialized
        self.mat = mat  
        self.in_dim = in_dim  
        self.out_dim = out_dim  
        
        #calculate
        self.num_hidden_nodes = self.mat.shape[1] - self.in_dim 
        
        #calculate total number of connection in matrix 
        self.connection_count = np.count_nonzero(self.mat)
        
        #when matrix has hidden layer  
        if self.num_hidden_nodes == 1:  
            self.hidden_dim = [1] 
        elif self.num_hidden_nodes == 0:
            self.hidden_dim = []
        else:
            self.hidden_dim = self.get_hidden_dim()   
            
            
    def get_hidden_dim(self):
        in_dim = self.in_dim
        out_dim = self.out_dim
        mat_mask = self.mat
        
        hidden_dim_list = []
        start_col_idx = 0
        finish_col_idx = in_dim -1   
        
        while(True):
            
            if finish_col_idx >= mat_mask.shape[1]:   
                print(finish_col_idx)
                break  
            
            if ((mat_mask.shape[0] - sum(hidden_dim_list)) == out_dim):  #example4 해결
                 break  #지금 hidden dimension들 합이랑 output dim 합이 row길이랑 같으면 더이상 탐색 필요 x
            
            for i in range(sum(hidden_dim_list), len(mat_mask)): #이부분이상한데..?   
    
                #밑에처럼 하면 example 2에서 오류가 남.
                #skip connection에 대한 예외처리 해줘야 함   
                if(mat_mask[i,start_col_idx:(finish_col_idx + 1)].sum() == 0):   
                
                    hidden_dim = i - sum(hidden_dim_list)
                    hidden_dim_list += [hidden_dim]
                    start_col_idx = finish_col_idx + 1
                    finish_col_idx += hidden_dim   
                    break    
                    
        return hidden_dim_list     
    
    
# wrapping activation function
# initialize constant_weight and bias    
def wrap_activation(layer, x, idx_activation, activations) :    
    if idx_activation == 0 :
        assert True
    elif idx_activation == 1 :
        return layer(x)
    else : 
        return activations[idx_activation](layer(x))
    
    
#### convert adjacency matrix to pytorch model  
# without hidden layer counts  
# search for all nodes connected to current node using dictionary of nodes  
# connecting and forward propagation simultaneously 
class SparseModel(nn.Module) : 
    def __init__(self, mat_wann, activations, constant_weight, output_activation) : 
        super(SparseModel, self).__init__()
        self.mat = mat_wann.mat
        self.in_dim = mat_wann.in_dim
        self.out_dim = mat_wann.out_dim
        self.num_hidden_nodes = mat_wann.num_hidden_nodes
        self.hidden_dim = mat_wann.hidden_dim
        
        self.activations = activations
        self.output_activation = output_activation
        #self.constant_weight = constant_weight
        
        self.nodes = {}
        '''
        nodes라는 dictionary 안에 아래와 같이 저장됨
        'hidden_1' : 해당 노드
        'hidden_2' : 해당 노드
        ...
        'output_1' : 해당 output 노드, hidden node로부터 연결되어있음
        'output_2' : 해당 output 노드, input node, hidden node로부터 연결되어있음
        '''
        self.connection_count = mat_wann.connection_count 
        
        # Reference
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/2
        # https://pytorch.org/docs/stable/nn.html
        #self.linears = nn.ModuleList([nn.Linear(1, 1, bias=False) for i in range(0, self.connection_count)])
        
        layer = nn.Linear(1, 1, bias=False)
        layer.weight.data.fill_(constant_weight)
        self.linears = nn.ModuleList([layer])
        for i in range(0, self.connection_count) : 
            layer = nn.Linear(1, 1, bias=False)
            layer.weight.data.fill_(constant_weight)
            self.linears.append(layer)

        
    def forward(self, x) : 
        
        # hidden node가 한개라도 있을때
        self.connect(x)
        
        # output은 반드시 있음
        outputs = self.concat_output()
        
        return outputs, self.nodes
    
    def concat_output(self) :
        for idx_output_node in list(range(self.out_dim)) :

            if idx_output_node == 0 :
                outputs = self.nodes['output_%d'%idx_output_node]
            else : 
                outputs = torch.cat((outputs, self.nodes['output_%d'%idx_output_node]), 1)
        
        return self.output_activation(outputs)
    
    def connect(self, x) : 
        # input layer와 모든 이전 hidden layer를 탐색
        # 그렇지 않으면 skip connection을 놓칠수 있음
        # 모든 node와 connection은 dictionary self.nodes에 저장
        #print(self.hidden_dim)
        hidden_node_counts = 0
        total_connection_counts = 0
        
        
        #hidden 노드가 없어도 이 코드가 돌아가도록  
        if self.num_hidden_nodes == 0:  
            
            ## input이랑 output만 이어주기
            for idx_output_row in range(self.mat.shape[0]): 
                
                connections_from_input = self.mat[idx_output_row,:]  
                if connections_from_input.sum() != 0:  
                    count_connection = 0 
                    input_node = None
                
                    for idx_input_col, activation_type in enumerate(connections_from_input): 
                        
                        if activation_type != 0 and count_connection == 0:  
                            layer = self.linears[total_connection_counts]
                            input_node = wrap_activation(layer, 
                                                         x[:, idx_input_col].view(-1,1), 
                                                         activation_type, 
                                                         self.activations)
                            count_connection += 1
                            total_connection_counts += 1
                        elif activation_type != 0 and count_connection != 0 :   
                            new_node = None
                            layer = self.linears[total_connection_counts]
                            new_node = wrap_activation(layer, 
                                                       x[:, idx_input_col].view(-1,1), 
                                                       activation_type, 
                                                       self.activations)  
                            count_connection += 1
                            total_connection_counts += 1
                            input_node = input_node + new_node  
                 
                    self.nodes['output_%d'%(idx_output_row)] = input_node  
                    
                else : # connection이 없어도 ouput node는 존재해야 함. 그 값은 0
                    self.nodes['output_%d'%(idx_output_row)] = torch.zeros((x.shape[0], 1), requires_grad=True)
            
            
        
        ############################### loop for hidden nodes + output nodes  
        else:
            
            for idx_hidden_row in list(range(0, self.mat.shape[0])) :   

                connections_from_input = self.mat[idx_hidden_row, :]

                if connections_from_input.sum() != 0 :  
                    count_connection = 0   
                    input_node = None   
                    ############################# loop for input nodes
                    for idx_input_col, activation_type in enumerate(connections_from_input) :

                        if activation_type != 0 and count_connection == 0:
                            # x[sample index, positional index for input]

                            # 1) idx_input_col 이 input에서 오는 경우
                            if idx_input_col < self.in_dim : 
                                layer = self.linears[total_connection_counts]
                                input_node = wrap_activation(layer, x[:, idx_input_col].view(-1, 1), 
                                                             activation_type, 
                                                             self.activations)
                                total_connection_counts += 1
                            # 2) idx_input_col이 hidden에서 오는 경우
                            elif idx_input_col >= self.in_dim : 
                                layer = self.linears[total_connection_counts]
                                input_node = wrap_activation(layer, self.nodes['hidden_%d'%(idx_input_col-self.in_dim)],
                                                             activation_type, 
                                                             self.activations)
                                total_connection_counts += 1

                            #print(input_node)
                            count_connection += 1
                        elif activation_type != 0 and count_connection != 0 :
                            # x[sample index, positional index for input]

                            new_node = None
                            # 1) idx_input_col 이 input에서 오는 경우
                            if idx_input_col < self.in_dim : 
                                layer = self.linears[total_connection_counts]
                                new_node = wrap_activation(layer, x[:, idx_input_col].view(-1, 1), 
                                                           activation_type, 
                                                           self.activations)
                                total_connection_counts += 1
                                
                            # 2) idx_input_col이 hidden에서 오는 경우
                            elif idx_input_col >= self.in_dim : 
                                layer = self.linears[total_connection_counts]
                                new_node = wrap_activation(layer, self.nodes['hidden_%d'%(idx_input_col-self.in_dim)],
                                                           activation_type,
                                                           self.activations)
                                total_connection_counts += 1

                            input_node = input_node + new_node

                            count_connection += 1
                            
                    # connect all input nodes to given hidden node
                    if idx_hidden_row < self.num_hidden_nodes : 
                        self.nodes['hidden_%d'%idx_hidden_row] = input_node 
                    else : 
                        self.nodes['output_%d'%(idx_hidden_row-self.num_hidden_nodes)] = input_node    
                else : # connection이 없어도 ouput node는 존재해야 함. 그 값은 0
                    if idx_hidden_row >= self.num_hidden_nodes : 
                        self.nodes['output_%d'%(idx_hidden_row-self.num_hidden_nodes)] = torch.zeros((x.shape[0],1), requirs_grad=True) 
            # sum all numbers of hidden nodes from this layer      
            hidden_node_counts += 1     

            
            



