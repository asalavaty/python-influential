#! usr/bin/env python3

# Import the requirements
import pandas as pd
import igraph
from .ranNorm import rangeNormalize
from .sir import simulate_sir
from random import seed
from statistics import mean
from .rank import rank_cal
from copy import deepcopy

# =============================================================================
#
#    Neighborhood Connectivity
#
# =============================================================================

def neighborhood_connectivity(graph, vertices = None, mode = "all", verbose = False):

    """
    This function calculates the neighborhood connectivity of input
    vertices and works with both directed and undirected networks.

    :param graph: the graph to be evaluated.
    :type graph: a graph (network) of the igraph class (igraph.Graph).

    :param vertices: a list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param mode: the mode of neighborhood connectivity depending on the
    directedness of the graph. If the graph is undirected, the mode "all"
    should be specified. Otherwise, for the calculation of neighborhood
    connectivity based on incoming connections select "in" and for
    the outgoing connections select "out". Also, if all of the connections
    are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and Neighborhood_connectivity

    """

    # Getting the names of vertices
    if vertices is None:
        vertices = graph.vs

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices
    
    if verbose:
        print("Getting the first neighbors of each node")

    # Getting the first neighbors of each node
    node_neighbors = graph.neighborhood(vertices = node_names, order = 1, mode = mode)

    if verbose:
        print("Removing the node index of each node from the list of its neighborhood")

    # Removing the node index of each node from the list of its neighborhood
    node_neighbors = list(map(lambda x:x[1:], node_neighbors))

    if verbose:
        print("Getting the neighborhood size of each neighbor")

    # Getting the neighborhood size of each neighbor
    first_neighbors_size  = list(map(lambda x: graph.neighborhood_size(vertices = x, order = 1, mode = mode), node_neighbors))

    if verbose:
        print("Reducing the count of node itself from the neighborhood size of each node")

    # Reducing the count of node itself from the neighborhood size of each node
    first_neighbors_size_final = list()
    for i in first_neighbors_size:
        first_neighbors_size_final.append(list(map(lambda x: x - 1, i)))
    
    del first_neighbors_size

    if verbose:
        print("Calculating the neighborhood connectivity of nodes")

    # Calculating the neighborhood connectivity of nodes
    first_neighbors_size_sum = list(map(sum, first_neighbors_size_final))
    first_neighbors_length = list(map(len, node_neighbors))
    temp_nc = [i / j for i, j in zip(first_neighbors_size_sum, first_neighbors_length)]
    
    # Preparing the results
    nc_table = pd.DataFrame({
        'Node_index':list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name':node_names,
        'Neighborhood_connectivity':temp_nc}, 
        index = None if len(node_names) > 1 else [0])

    # Handling missing results
    nc_table.fillna(0)
    
    return(nc_table)

#=============================================================================
#
#    H-index
#
#=============================================================================

def h_index(graph, vertices = None, mode = "all", verbose = False):

    """
    This function calculates the H-index of input vertices and
    works with both directed and undirected networks.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).
    
    :param vertices: A list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param mode: The mode of H-index depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of H-index based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool
    
    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and H-index.

    """

    # Getting the names of vertices
    if vertices is None:
        vertices = graph.vs

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    if verbose:
        print("Getting the first neighbors of each node")

    # Getting the first neighbors of each node
    first_neighbors = graph.neighborhood(vertices = node_names, order = 1, mode = mode)

    if verbose:
        print("Removing the node index of each node from the list of its neighborhood")

    # Removing the node index of each node from the list of its neighborhood
    first_neighbors = list(map(lambda x:x[1:], first_neighbors))

    if verbose:
        print("Getting the neighborhood size of each node")

    # Getting the neighborhood size of each node
    first_neighbors_size = list(map(lambda x: graph.neighborhood_size(vertices = x, order =1, mode = mode), first_neighbors))

    if verbose:
        print("Reducing the count of node itself from the neighborhood size of each node")

    # Reduce the count of node itself from neighborhood size
    first_neighbors_size_final = list()
    for i in first_neighbors_size:
        first_neighbors_size_final.append(list(map(lambda x: x - 1, i)))
    
    del first_neighbors_size

    # Calculation of H-index

    if verbose:
        print("Calculating the H-index")

    ## Descending sort the neighbors size
    first_neighbors_size_final = list(map(lambda x: sorted(x, reverse=True), first_neighbors_size_final))

    ## Get the H-index
    tmp_h_index = list()

    for i in first_neighbors_size_final:
        if len(i) == 0:
            tmp_h_index.append(0)
        elif max(i) == 0:
            tmp_h_index.append(0)
        else:
            tmp_value = list(map(lambda x,y: x>=y, i, list(range(1, len(i)+1))))
            tmp_value_index = list(index for index , val in enumerate(tmp_value) if val)
            tmp_value_index = list(map(lambda x: x+1, tmp_value_index))
            tmp_h_index.append(tmp_value_index[len(tmp_value_index)-1])
    
    # Preparing the results
    h_index_table = pd.DataFrame({
        'Node_index':list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name':node_names,
        'H_index':tmp_h_index
        }, index= None if len(node_names) > 1 else [0])
    
    # Handling missing results
    h_index_table.fillna(0)

    return h_index_table

#=============================================================================
#
#    Local H-index
#
#=============================================================================

def lh_index(graph, vertices = None, mode = "all", verbose = False):

    """
    This function calculates the local H-index of input vertices and
    works with both directed and undirected networks.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).

    :param vertices: A list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param mode: The mode of local H-index depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of local H-index based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and LH_index.

    """

    # Getting the names of vertices
    if vertices is None:
        vertices = graph.vs
    
    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices
    
    # Getting the first neighbors of each node
    first_neighbors = graph.neighborhood(vertices = node_names, order = 1, mode = mode)

    # Calculation of local H-index (LH-index)
    tmp_lh_index = list(map(lambda x: sum(h_index(graph=graph, vertices=graph.vs[x], mode = mode, verbose = verbose).H_index), first_neighbors))

    if verbose:
        print("Preparing the LH-index")

    # Preparing the results
    lh_index_table = pd.DataFrame({
        'Node_index':list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name':node_names,
        'LH_index':tmp_lh_index
    }, index= None if len(node_names) > 1 else [0])

    # Handling the missing values
    lh_index_table.fillna(0)

    return lh_index_table

#=============================================================================
#
#    Collective Influence (CI)
#
#=============================================================================

def collective_influence(graph, vertices =None, mode = "all", d=3, verbose = False):

    """
    This function calculates the collective influence of input vertices and
    works with both directed and undirected networks. This function and its descriptions are
    obtained from https://github.com/ronammar/collective_influence with minor modifications.
    Collective Influence as described by Morone & Makse (2015). In simple terms,
    it is the product of the reduced degree (degree - 1) of a node and the total reduced
    degree of all nodes at a distance d from the node.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).

    :param vertices: A list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param mode: The mode of collective influence depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of collective influence based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param d: The distance, expressed in number of steps from a given node (default=3). Distance
    must be > 0. According to Morone & Makse (https://doi.org/10.1038/nature14604), optimal
    results can be reached at d=3,4, but this depends on the size/radius of the network.
    Note: the distance d is not inclusive. This means that nodes at a distance of 3 from
    our node-of-interest do not include nodes at distances 1 and 2. Only 3.
    :type d: int

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and Collective_influence.

    """

    # Getting the names of vertices
    if vertices is None:
        vertices = graph.vs
    
    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    if verbose:
        print("Calculating the reduced degrees of nodes")

    # Calculate the reduced degree
    reduced_degrees = graph.degree(vertices = node_names, mode = mode)
    reduced_degrees = list(map(lambda x: x - 1, reduced_degrees))

    if verbose:
        print("Identifying only neighbors at distance d")

    # Identify only neighbors at distance d
    nodes_at_distance = graph.neighborhood(vertices = node_names, mode = mode, order = d, mindist = d)

    if verbose:
        print("Calculating the reduced degrees of neighbors at distance d")

    # Calculate the reduced degree of neighbors at distance d
    nodes_at_distance_indices = list(set([i for tmp_list in nodes_at_distance for i in tmp_list])) # We use set() to get only unique values from the list and reconvert to list
    nodes_at_distance_reduced_degrees = graph.degree(vertices = graph.vs[nodes_at_distance_indices], mode = mode)
    nodes_at_distance_reduced_degrees = list(map(lambda x: x - 1, nodes_at_distance_reduced_degrees))

    # Calculate the Collective Influence

    ## get the reduced degrees of neighbors at distance d
    reduced_degrees_neighbors = list(map(lambda x: [nodes_at_distance_reduced_degrees[i] for i in [nodes_at_distance_indices.index(j) for j in x]], nodes_at_distance))
    reduced_degrees_neighbors_sum = list(map(lambda x: sum(x), reduced_degrees_neighbors)) # sum up the reduced_degrees_neighbors

    if verbose:
        print("Calculating the Collective Influence")

    ## calculate the ci
    tmp_ci = list(map(lambda x, y: x * y, reduced_degrees, reduced_degrees_neighbors_sum))

    # Preparing the results
    ci_table = pd.DataFrame({
        'Node_index':list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name':node_names,
        'Collective_influence':tmp_ci
    }, index= None if len(node_names) > 1 else [0])

    # Handling missing values
    ci_table.fillna(0)

    return ci_table

#=============================================================================
#
#    ClusterRank (CR)
#
#=============================================================================

def clusterRank(graph, vertices = None, directed = False, loops = True, verbose = False):

    """
    This function calculates the ClusterRank of input vertices and
    works with both directed and undirected networks.
    This function and all of its descriptions have been adapted from the centiserve R package with
    some minor modifications. ClusterRank is a local ranking algorithm which takes into account not only
    the number of neighbors and the neighbors’ influences, but also their clustering coefficient.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).

    :param vertices: A list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param directed: Whether a directed graph is analyzed. This argument is ignored for undirected graphs.
    :type directed: bool

    :param loops: Whether the loop edges are also counted.
    type loops: bool

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and ClusterRank.

    """
    # Getting the names of vertices
    if vertices is None:
        vertices = graph.vs

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    # Define the mode of clusterRank calculation
    if directed:
        clusterRank_mode = "out"
    else:
        clusterRank_mode = "all"

    if verbose:
        print("Getting the first neighbors of each node")

    # Getting the first neighbors of each node
    first_neighbors = graph.neighborhood(vertices = node_names, order = 1, mode = clusterRank_mode)

    # Removing the node index of each node from the list of its neighborhood
    first_neighbors = list(map(lambda x: x[1:], first_neighbors))

    if verbose:
        print("Calculating the node transitivity")

    # Calculate the vertex transitivity
    vertex_transitivity = list()
    if clusterRank_mode == "all":
        vertex_transitivity = graph.transitivity_local_undirected(vertices = node_names, mode = 'zero')
    else:
        for i in first_neighbors:
            if len(i) < 2:
                vertex_transitivity.append(0)
            else:
                tmp_subgraph = graph.subgraph(i)
                vertex_transitivity.append(tmp_subgraph.ecount()/tmp_subgraph.vcount()*(tmp_subgraph.vcount()-1))

    if verbose:
        print("Calculating the initial ClusterRank value")

    # Calculate the initial clusterRank value
    tmp_clRank_list = list()
    for i in first_neighbors:
        tmp_clRank = 0
        for j in i:
            tmp_clRank = tmp_clRank + graph.degree(vertices = j, mode = clusterRank_mode, loops = loops) + 1
        tmp_clRank_list.append(tmp_clRank)

    if verbose:
        print("Calculating the final ClusterRank value")

    # Calculate the final ClusterRank
    final_clusterRank = list(map(lambda x, y: x * y, tmp_clRank_list, vertex_transitivity))

    # Preparing the results
    cr_table = pd.DataFrame({
        'Node_index':list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name':node_names,
        'ClusterRank':final_clusterRank
    }, index= None if len(node_names) > 1 else [0])

    return cr_table

#=============================================================================
#
#    IVI from indices
#
#=============================================================================

def ivi_from_indices(DC, CR, LH_index, NC, BC, CI, scaled = True, verbose = False):

    """
    This function calculates the IVI of the desired nodes from previously calculated centrality
    measures. This function is not dependent to other packages and the required centrality
    measures, namely degree centrality, ClusterRank, betweenness centrality, Collective Influence,
    local H-index, and neighborhood connectivity could have been calculated by any means beforehand.
    A shiny app has also been developed for the calculation of IVI as well as IVI-based network
    visualization, which is accessible online at https://influential.erc.monash.edu/.

    :param DC: A list containing the values of degree centrality of the desired vertices.
    :type DC: list

    :param CR: A list containing the values of ClusterRank of the desired vertices.
    :type CR: list

    :param LH_index: A list containing the values of local H-index of the desired vertices.
    :type LH_index: list

    :param NC: A list containing the values of neighborhood connectivity of the desired vertices.
    :type NC: list

    :param BC: A list containing the values of betweenness centrality of the desired vertices.
    :type BC: list

    :param CI: A list containing the values of Collective Influence of the desired vertices.
    :type CI: list

    :param: scaled: Wether the end result should be 1-100 range normalized or not (default is True).
    :type scaled: bool

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A numeric list with the IVI values based on the provided centrality measures.

    """

    # Prepare a temporary table of the input indices
    input_indices = pd.DataFrame({
        'DC': DC, 
        'CR': CR, 
        'LH_index': LH_index, 
        'NC': NC, 
        'BC': BC, 
        'CI': CI
    }, index= None if len(DC) > 1 else [0])

    # Removing the nan and na values
    input_indices.fillna(0)

    if verbose:
        print("1-100 normalization of centrality measures")

    #1-100 normalization of centrality measures
    for i in list(range(len(input_indices.columns))):
        if(any(list(map(lambda x: x > 0, input_indices.iloc[:,i])))):
            input_indices.iloc[:,i] = rangeNormalize(data = input_indices.iloc[:,i], minimum=1, maximum=100)

    # Calculation of IVI

    if verbose:
        print("Calcutaion of the Spreading Rank")

    ## Calcutaion of the Spreading Rank
    spreading_rank = ((input_indices.NC+input_indices.CR)*(input_indices.BC+input_indices.CI))

    if any(list(map(lambda x: x == 0 or pd.isna(x), spreading_rank))):
        spreading_rank[((spreading_rank == 0) + pd.isna(spreading_rank))] = 1

    if verbose:
        print("Calcutaion of the Hubness Rank")

    ## Calcutaion of the Hubness Rank
    hubness_rank = (input_indices.DC+input_indices.LH_index)

    if any(list(map(lambda x: x == 0 or pd.isna(x), hubness_rank))):
        hubness_rank[((hubness_rank == 0) + pd.isna(hubness_rank))] = 1

    if verbose:
        print("Calculating the IVI")

    ## Calculating the IVI itself
    tmp_ivi = hubness_rank * spreading_rank

    # 1-100 normalization of IVI
    if scaled:
        if verbose:
            print("1-100 normalization of IVI")
        if len(tmp_ivi) > 1:
            tmp_ivi = rangeNormalize(data = tmp_ivi, minimum=1, maximum=100)

    return tmp_ivi

#=============================================================================
#
#    IVI from a graph
#
#=============================================================================

def ivi(graph, vertices = None, weights = None, directed = False, mode = "all", loops = True, d = 3, scaled = True, verbose = False):

    """

    This function calculates the IVI of the desired nodes from a graph.
    A shiny app has also been developed for the calculation of IVI as well as IVI-based network
    visualization, which is accessible online at https://influential.erc.monash.edu/.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).

    :param vertices: A list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param weights: Optional positive weight list for calculating weighted betweenness centrality
    of nodes as a requirement for calculation of IVI. If the graph has a weight edge attribute,
    then this is used by default. Can be a sequence or iterable or even an edge attribute name. 
    Weights are used to calculate weighted shortest paths, so they are interpreted as distances.
    :type weights: list

    :param directed: Whether the graph is analyzed as a directed graph. This argument is ignored for undirected graphs.
    :type directed: bool

    :param mode: The mode of IVI depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of IVI based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param loops: Whether the loop edges are also counted.
    :type loops: bool

    :param d: The distance, expressed in number of steps from a given node (default=3). Distance
    must be > 0. According to Morone & Makse (https://doi.org/10.1038/nature14604), optimal
    results can be reached at d=3,4, but this depends on the size/radius of the network.
    Note: the distance d is not inclusive. This means that nodes at a distance of 3 from
    our node-of-interest do not include nodes at distances 1 and 2. Only 3.
    :type d: int

    :param: scaled: Wether the end result should be 1-100 range normalized or not (default is True).
    :type scaled: bool

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: A Pandas DataFrame with three columns including Node_index, Node_name, and IVI.

    """

    # Preparing the node names
    if vertices is None:
        vertices = graph.vs
    
    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    # Calculation of required centrality measures
    if verbose:
        print("Calculating the Degree Centrality of Nodes")

    tmp_DC = graph.degree(vertices = vertices, mode = mode, loops = loops)

    if verbose:
        print("Calculating the ClusterRank of Nodes")
   
    tmp_CR = clusterRank(graph = graph, vertices = vertices, directed = directed, loops = loops, verbose = verbose)

    if verbose:
        print("Calculating the Local H-index of Nodes")

    tmp_LH_index = lh_index(graph = graph, vertices = vertices, mode = mode, verbose = verbose)

    if verbose:
        print("Calculating the Neighborhood Connectivity of Nodes")

    tmp_NC = neighborhood_connectivity(graph = graph, vertices = vertices, mode = mode, verbose = verbose)

    if verbose:
        print("Calculating the Betweenness Centrality of Nodes")

    tmp_BC = graph.betweenness(vertices = vertices, directed = directed, weights = weights)

    if verbose:
        print("Calculating the Collective Influence of Nodes")

    tmp_CI = collective_influence(graph = graph, vertices = vertices, mode = mode, d = d, verbose = verbose)

    # Prepare a temporary table of the input indices
    input_indices = pd.DataFrame({
        'DC': tmp_DC,
        'CR': tmp_CR.ClusterRank,
        'LH_index': tmp_LH_index.LH_index,
        'NC': tmp_NC.Neighborhood_connectivity,
        'BC': tmp_BC,
        'CI': tmp_CI.Collective_influence
    }, index= None if len(node_names) > 1 else [0])

    # Remove na and nan from the DataFrame
    input_indices.fillna(0)

    if verbose:
        print("1-100 normalization of centrality measures")

    #1-100 normalization of centrality measures
    for i in list(range(len(input_indices.columns))):
        if(any(list(map(lambda x: len(input_indices.iloc[:,i]) > 1 and x > 0, input_indices.iloc[:,i])))):
            input_indices.iloc[:,i] = rangeNormalize(data= input_indices.iloc[:,i], minimum=1, maximum=100)

    if verbose:
        print("Calculating the Spreading Rank")

    ## Calcutaion of the Spreading Rank
    spreading_rank = ((input_indices.NC+input_indices.CR)*(input_indices.BC+input_indices.CI))

    if any(list(map(lambda x: x == 0 or pd.isna(x), spreading_rank))):
        spreading_rank[((spreading_rank == 0) + pd.isna(spreading_rank))] = 1

    if verbose:
        print("Calculating the Hubness Rank")

    ## Calcutaion of the Hubness Rank
    hubness_rank = (input_indices.DC+input_indices.LH_index)

    if any(list(map(lambda x: x == 0 or pd.isna(x), hubness_rank))):
        hubness_rank[((hubness_rank == 0) + pd.isna(hubness_rank))] = 1

    if verbose:
        print("Calculating the IVI")

    ## Calculating the IVI itself
    tmp_ivi = hubness_rank * spreading_rank

    # 1-100 normalization of IVI
    if scaled:
        if verbose:
            print("1-100 normalization of IVI")
        if len(tmp_ivi) > 1:
            tmp_ivi = rangeNormalize(data= tmp_ivi, minimum=1, maximum=100)

    # Preparing the results
    ivi_table = pd.DataFrame({
        'Node_index': list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name': node_names,
        'IVI': tmp_ivi
    })

    return ivi_table

#=============================================================================
#
#    Spreading score
#
#=============================================================================

def spreading_score(graph, vertices = None, weights = None, directed = False, mode = "all", loops = True, d = 3, scaled = True, verbose = False):

    """

    This function calculates the Spreading score of the desired nodes from a graph.
    Spreading score reflects the spreading potential of each node within a network and is
    one of the major components of the IVI.

    :param graph: the graph to be evaluated.
    :type graph: a graph (network) of the igraph class (igraph.Graph).

    :param vertices: a list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param weights: optional positive weight list for calculating weighted betweenness centrality
    of nodes as a requirement for calculation of Spreading score. If the graph has a weight edge attribute,
    then this is used by default. Can be a sequence or iterable or even an edge attribute name. 
    Weights are used to calculate weighted shortest paths, so they are interpreted as distances.
    :type weights: list

    :param directed: whether the graph is analyzed as a directed graph. This argument is ignored for undirected graphs.
    :type directed: bool

    :param mode: the mode of Spreading score depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of Spreading score based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    :param loops: whether the loop edges are also counted.
    :type loops: bool

    :param d: the distance, expressed in number of steps from a given node (default=3). Distance
    must be > 0. According to Morone & Makse (https://doi.org/10.1038/nature14604), optimal
    results can be reached at d=3,4, but this depends on the size/radius of the network.
    Note: the distance d is not inclusive. This means that nodes at a distance of 3 from
    our node-of-interest do not include nodes at distances 1 and 2. Only 3.
    :type d: int

    :param: scaled: Wether the end result should be 1-100 range normalized or not (default is True).
    :type scaled: bool

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: a Pandas DataFrame with three columns including Node_index, Node_name, and Spreading_score.

    """

    # Preparing the node names
    if vertices is None:
        vertices = graph.vs

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    # Calculation of required centrality measures

    if verbose:
        print("Calculating the ClusterRank of Nodes")

    tmp_CR = clusterRank(graph = graph, vertices = vertices, directed = directed, loops = loops, verbose = verbose)

    if verbose:
        print("Calculating the Neighborhood Connectivity of Nodes")

    tmp_NC = neighborhood_connectivity(graph = graph, vertices = vertices, mode = mode, verbose = verbose)

    if verbose:
        print("Calculating the Betweenness Centrality of Nodes")

    tmp_BC = graph.betweenness(vertices = vertices, directed = directed, weights = weights)

    if verbose:
        print("Calculating the Collective Influence of Nodes")
    tmp_CI = collective_influence(graph = graph, vertices = vertices, mode = mode, d = d, verbose = verbose)

    # Prepare a temporary table of the input indices
    input_indices = pd.DataFrame({
        'CR': tmp_CR.ClusterRank,
        'NC': tmp_NC.Neighborhood_connectivity,
        'BC': tmp_BC,
        'CI': tmp_CI.Collective_influence
    }, index= None if len(node_names) > 1 else [0])

    # Remove na and nan from the DataFrame
    input_indices.fillna(0)

    if verbose:
        print("1-100 normalization of centrality measures")

    #1-100 normalization of centrality measures
    for i in list(range(len(input_indices.columns))):
        if any(list(map(lambda x: len(input_indices.iloc[:,i]) > 1 and x > 0, input_indices.iloc[:,i]))):
            input_indices.iloc[:,i] = rangeNormalize(data= input_indices.iloc[:,i], minimum=1, maximum=100)

    # Calcutaion of the Spreading Rank
    spreading_rank = ((input_indices.NC+input_indices.CR)*(input_indices.BC+input_indices.CI))

    # 1-100 normalization of Spreading Rank
    if scaled:
        if verbose:
            print("1-100 normalization of Spreading Rank")
        if len(spreading_rank) > 1:
            spreading_rank = rangeNormalize(data=spreading_rank, minimum=1, maximum=100)

    # Preparing the results
    spreading_rank_table = pd.DataFrame({
        'Node_index': list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name': node_names,
        'Spreading_score': spreading_rank
    }, index= None if len(node_names) > 1 else [0])

    return spreading_rank_table

#=============================================================================
#
#    Hubness score
#
#=============================================================================

def hubness_score(graph, vertices = None, directed = False, mode = "all", loops = True, scaled = True, verbose = False):

    """

    This function calculates the Hubness score of the desired nodes from a graph.
    Hubness score reflects the power of each node in its surrounding environment and is
    one of the major components of the IVI.

    :param graph: the graph to be evaluated.
    :type graph: a graph (network) of the igraph class (igraph.Graph).

    :param vertices: a list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param directed: whether the graph is analyzed as a directed graph. This argument is ignored for undirected graphs.
    :type directed: bool

    :param mode: the mode of Hubness score depending on the directedness of the graph.
    If the graph is undirected, the mode "all" should be specified.
    Otherwise, for the calculation of Hubness score based on
    incoming connections select "in" and for the outgoing connections select "out".
    Also, if all of the connections are desired, specify the "all" mode. Default mode is set to "all".
    :type mode: string

    param loops: Whether the loop edges are also counted.
    :type loops: bool

    :param: scaled: Wether the end result should be 1-100 range normalized or not (default is True).
    :type scaled: bool

    :param verbose: whether the accomplishment of different stages of the algorithm should be printed (default is False).
    :type verbose: bool

    :return: a Pandas DataFrame with three columns including Node_index, Node_name, and Hubness_score.

    """

     # Preparing the node names
    if vertices is None:
        vertices = graph.vs

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names = [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    # Calculation of required centrality measures

    if verbose:
        print("Calculating the Degree Centrality of Nodes")

    tmp_DC = graph.degree(vertices = vertices, mode = mode, loops = loops)

    if verbose:
        print("Calculating the Local H-index of Nodes")

    tmp_LH_index = lh_index(graph = graph, vertices = vertices, mode = mode, verbose = verbose)

    # Prepare a temporary table of the input indices
    input_indices = pd.DataFrame({
        'DC': tmp_DC,
        'LH_index': tmp_LH_index.LH_index
    }, index= None if len(node_names) > 1 else [0])

    # Remove na and nan from the DataFrame
    input_indices.fillna(0)

    if verbose:
        print("1-100 normalization of centrality measures")

    # 1-100 normalization of centrality measures
    for i in list(range(len(input_indices.columns))):
        if(any(list(map(lambda x: len(input_indices.iloc[:,i]) > 1 and x > 0, input_indices.iloc[:,i])))):
            input_indices.iloc[:,i] = rangeNormalize(data= input_indices.iloc[:,i], minimum=1, maximum=100)

    ## Calcutaion of the Hubness Rank
    hubness_rank = (input_indices.DC+input_indices.LH_index)

    # 1-100 normalization of Hubness Rank
    if scaled:
        if verbose:
            print("1-100 normalization of Hubness Rank")
        if len(hubness_rank) > 1:
            hubness_rank = rangeNormalize(data=hubness_rank, minimum=1, maximum=100)

    # Preparing the results
    hubness_rank_table = pd.DataFrame({
        'Node_index': list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name': node_names,
        'Hubness_score': hubness_rank
    }, index= None if len(node_names) > 1 else [0])

    return hubness_rank_table


#=============================================================================
#
#    SIRIR
#
#=============================================================================

def sirir(graph, vertices = None, beta = 0.5, gamma = 0.1, no_sim = None,  model_seed = 1234, verbose = True):

    """

    This function is achieved by the integration susceptible-infected-recovered (SIR) model
    with the leave-one-out cross validation technique and ranks network nodes based on their
    true universal influence. One of the applications of this function is the assessment of
    performance of a novel algorithm in identification of network influential nodes by considering
    the SIRIR ranks as the ground truth (gold standard).

    :param graph: the graph to be evaluated.
    :type graph: a graph (network) of the igraph class (igraph.Graph).

    :param vertices: a list of desired vertices, which could be
    obtained by the igraph.VertexSeq function. You may also provide a list of string names of vertices.
    All the vertices are considered by default (vertices = None).
    :type vertices: igraph Vertex class or string list

    :param beta: the rate of infection of an individual that is susceptible
    and has a single infected neighbor. The infection rate of a susceptible individual with n
    infected neighbors is n times beta. Formally this is the rate parameter of an exponential
    distribution.
    :type beta: non-negative float


    :param gamma: the rate of recovery of an infected individual.
    Formally, this is the rate parameter of an exponential distribution.
    :type gamma: Non-negative float

    :param no_sim: The number of simulation runs to perform SIR model on the
    original network as well as perturbed networks generated by leave-one-out technique.
    You may choose a different no_sim based on the available memory on your system. Default (None)
    is set to number of vertices of the original graph multiplied by 100.
    :type no_sim: int

    :param seed: a single value, interpreted as an integer to be used for random number generation.
    :type seed: int

    :return: a Pandas DataFrame with four columns including Node_index, Node_name, the difference values of the original and
    perturbed networks for each node (Difference_value), and a column containing node influence rankings (Rank).

    """

    # Preparing the node names
    if vertices is None:
        vertices = graph.vs

    # Preparing the no_sim
    if no_sim is None:
        no_sim = graph.vcount()*100

    if isinstance(vertices, igraph.VertexSeq):
        node_names = vertices['name']
    elif isinstance(vertices, igraph.Vertex):
        node_names =  [vertices['name']]
    elif isinstance(vertices, str):
        node_names = [vertices]
    else:
        node_names = vertices

    # Prepare a container for the temp_loocr
    temp_loocr_diff = []

    # Model the spreading based on all nodes
    seed(model_seed)
    all_included_spread = simulate_sir(graph= graph, beta= beta, gamma= gamma, no_sim= no_sim)

    # Getting the mean of spread in each independent experiment
    all_mean_spread = []
    for i in all_included_spread:
        all_mean_spread.append(max(i['Recovered']))
    
    all_mean_spread = mean(all_mean_spread)

    if verbose:
        print("The SIR simulation of the original network is done!")

    # Model the spread based on leave one out cross ranking (LOOCR)
    for s in range(len(node_names)):
        temp_graph = deepcopy(graph)
        temp_graph.delete_vertices(node_names[s])
        seed(model_seed)

        loocr_spread = simulate_sir(graph = temp_graph, beta = beta, gamma = gamma, no_sim = no_sim)

        loocr_mean_spread = []

        for h in loocr_spread:
            loocr_mean_spread.append(max(i['Recovered']))

        loocr_mean_spread = mean(loocr_mean_spread)

        ## Calculate the difference between spread in the original network and the purturbed one
        temp_loocr_diff.append(all_mean_spread - loocr_mean_spread)

        if verbose:
            print("The SIR simulation after removing %s is done!" % (node_names[s]))

    # Rank temp_loocr_diff values
    temp_loocr_rank = rank_cal(list(map(lambda x: x * -1, temp_loocr_diff)))

    # Preparing the results
    sirir_tbl = pd.DataFrame({
        'Node_index': list(map(lambda x: graph.vs['name'].index(x), node_names)),
        'Node_name': node_names,
        'Difference_value': temp_loocr_diff,
        'Rank': temp_loocr_rank
    }, index=None if len(node_names) > 1 else [0])

    return(sirir_tbl)
