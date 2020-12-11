def test_generate_sequential_wiring():
    import graphtorch as gt
    from graphtorch.wiring import connect_sequential

    in_dim = 2
    hidden_node_dims = [2, 4, 3]
    out_dim = 3
    graph = gt.Graph(in_dim, out_dim, hidden_node_dims)

    graph = connect_sequential(graph)
    
def test_generate_parallel_wiring():
    import graphtorch as gt
    from graphtorch.wiring import connect_parallel

    in_dim = 2
    hidden_node_dims = [3, 4, 5, 6, 7]
    out_dim = 10
    graph = gt.Graph(in_dim, out_dim, hidden_node_dims)

    nodes_per_hidden_layer = [2, 3]
    graph = connect_parallel(graph, nodes_per_hidden_layer)


def test_generate_random_wiring():
    import graphtorch as gt
    from graphtorch.wiring import connect_randomly

    in_dim = 2
    hidden_node_dims = [3, 4, 5, 6, 7]
    out_dim = 10
    graph = gt.Graph(in_dim, out_dim, hidden_node_dims)

    seed = 0
    network_sparsity = 0.1
    graph = connect_randomly(graph, seed, network_sparsity)
