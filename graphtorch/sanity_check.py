def wires_sanity_check(connection, dimension):
    sanity_checks = [
        connection_ratio_check,
        forward_feasibility_check,
        broken_path_check,
        output_feasibility_check,
    ]
    results = [f(connection, dimension) for f in sanity_checks]
    return results


def connection_ratio_check(connection, dimension):
    """
    """
    return ["Connection ratio check", True]


def forward_feasibility_check(connection, dimension):
    """ 
    """
    return ["Forward reasibility check", True]


def broken_path_check(connection, dimension):
    """ 
    """
    return ("Broken path check", True)


def output_feasibility_check(connection, dimension):
    """
    """
    return ("Ouput feasibility check", True)


def get_froms(connection, node_to):
    return connection[node_to][connection[node_to] == 1].index.tolist()


def get_previous_nodes_dict(connection):
    node_keys = connection.columns.tolist()
    previous_nodes_dict = {}
    for node_key in node_keys:
        previous_nodes_dict[node_key] = get_froms(connection, node_key)
    return previous_nodes_dict


def get_depths(connection):
    previous_nodes_dict = get_previous_nodes_dict(connection)
