import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import networkx as nx
import scipy.sparse


def weights_init(m):
    """Initialize neural network
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_uniform_(m.weight)  # Glorot uniform initialization
        if m.bias is not None:
            init.zeros_(m.bias)         # Initialize biases to zero

def _incidence_matrix(  # noqa: C901
    graph,
    nodelist=None,
    edgelist=None,
    oriented=False,
    weight=None,
    dtype=np.float_
):  # pragma: no cover
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the tail of an edge and 1
    for the head of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    graph : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       Default weight None is equivelent to 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    Returns
    -------
    A : SciPy sparse matrix
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://academicearth.org/lectures/network-applications-incidence-matrix
    """
    if nodelist is None:
        nodelist = list(graph)
    if edgelist is None:
        if graph.is_multigraph():
            edgelist = list(graph.edges(keys=True))
        else:
            edgelist = list(graph.edges())

    # Prepare matrix
    rows = []
    cols = []
    data = []
    node_index = {node: i for i, node in enumerate(nodelist)}
    # Populate rows and cols
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as exc:
            raise nx.NetworkXError(f"node {u} or {v} in edgelist but not in nodelist") from exc
        if weight is None:
            wt = 1
        else:
            if graph.is_multigraph():
                ekey = e[2]
                wt = graph[u][v][ekey].get(weight, 1)
            else:
                wt = graph[u][v].get(weight, 1)
        if oriented:
            rows.append(ui)
            cols.append(ei)
            data.append(-wt)
            rows.append(vi)
            cols.append(ei)
            data.append(wt)
        else:
            rows.append(ui)
            cols.append(ei)
            data.append(wt)
            rows.append(vi)
            cols.append(ei)
            data.append(wt)
    # Create sparse adjacency matrix
    matric = scipy.sparse.coo_matrix(
        (np.array(data, dtype=dtype), (np.array(rows), np.array(cols))),
        shape=(len(nodelist), len(edgelist)),
    )
    return matric


# End imported code


def n2n_construct(graph):
    """Convert graph to node-node adjancency matrix.

    Args:
        graph (Networkx.Graph): graph built by Networkx.

    Returns:
        scipy.sparse.matrix: sparse matrix of adjacency matrix.
    """
    return nx.to_scipy_sparse_matrix(graph, weight=None, dtype=np.float32, format="coo")

def e2n_construct(graph):
    """Convert graph to edge-node matrix.

    Args:
        graph (Networkx.Graph): graph built by Networkx.

    Returns:
        scipy.sparse.matrix: sparse matrix of node-edge relation.
    """
    return _incidence_matrix(graph, weight=None, dtype=np.float32, format="coo")

def subgraph_construct(graph, graph_list):
    """Construct matrix of subgraph.

    Args:
        graph (Networkx.Graph): complete graph built by Networkx
        graph_list (List[List[str]]): list of subgraph. each subgraph is represented by a list of included nodes.
    """
    nodelist = list(graph.nodes)
    num_nodes = len(nodelist)
    num_subgraphs = len(graph_list)

    node_to_index = {node: i for i, node in enumerate(nodelist)}

    # Prepare row, col, and data for COO format
    rows = []
    cols = []
    data = []

    # Iterate over each subgraph and mark presence of nodes
    for subgraph_idx, subgraph in enumerate(graph_list):
        for node in subgraph:
            rows.append(subgraph_idx)            # Subgraph index as row
            cols.append(node_to_index[node])      # Node index as column
            data.append(1)                        # Presence indicated by 1

    return rows, cols, data, (num_subgraphs, num_nodes)

def prepare_mean_field(graph, graph_list):
    """prepare matrixes for mean field embedding.

    Args:
        graph (Networkx.Graph): complete graph built by Networkx
        graph_list (List[List[str]]): list of subgraph. each subgraph is represented by a list of included nodes.

    Returns:
        n2n matrix, e2n matrix, subgraph matrix (all sparse version)
    """

    # Create a PyTorch sparse COO tensor for n2n
    n2n_matrix = _incidence_matrix(graph)
    n2n_indices = torch.tensor([n2n_matrix.row, n2n_matrix.col], dtype=torch.long)
    n2n_values = torch.tensor(n2n_matrix.data, dtype=torch.float32)
    shape = n2n_matrix.shape
    n2n_sp = torch.sparse_coo_tensor(n2n_indices, n2n_values, torch.Size(shape))

    # Create a PyTorch sparse COO tensor for e2n
    e2n_matrix = e2n_construct(graph)
    e2n_indices = torch.tensor([e2n_matrix.row, e2n_matrix.col], dtype=torch.long)
    e2n_values = torch.tensor(e2n_matrix.data, dtype=torch.float32)
    shape = e2n_matrix.shape
    e2n_sp = torch.sparse_coo_tensor(e2n_indices, e2n_values, torch.Size(shape))

    # Creat a PyTorch sparse COO tensor for subgraph
    subg_sp_rows, subg_sp_cols, subg_sp_data, (num_subgraphs, num_nodes) = subgraph_construct(graph, graph_list)
    subg_indices = torch.tensor([subg_sp_rows, subg_sp_cols], dtype=torch.long)
    subg_values = torch.tensor(subg_sp_data, dtype=torch.float32)
    subg_shape = (num_subgraphs, num_nodes)
    subg_sp = torch.sparse_coo_tensor(subg_indices, subg_values, torch.Size(subg_shape))   

    return n2n_sp, e2n_sp, subg_sp