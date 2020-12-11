def test_create_graph() : 
    import graphtorch as gt
    in_dim = 2
    hidden_node_dims = [2, 4, 3]
    out_dim = 3
    graph = gt.Graph(in_dim, out_dim, hidden_node_dims)
    
def test_create_graph_in_out_splitted() : 
    import graphtorch as gt
    in_dim = 2
    hidden_node_dims = [2, 4, 3]
    out_dim = 3
    graph = gt.Graph(
        in_dim, out_dim, hidden_node_dims, split_input_layer=True, split_output_layer=True
    )

def test_generate_torch_object_from_graph() : 
    import graphtorch as gt
    in_dim = 2
    hidden_node_dims = [2, 4, 3]
    out_dim = 3
    graph = gt.Graph(
        in_dim, out_dim, hidden_node_dims, split_input_layer=True, split_output_layer=True
    )
    layer = gt.GraphLayer(graph)