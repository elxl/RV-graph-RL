import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import networkx as nx
import scipy.sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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
    return nx.to_scipy_sparse_array(graph, weight=None, dtype=np.float32, format="coo")

def e2n_construct(graph):
    """Convert graph to edge-node matrix.

    Args:
        graph (Networkx.Graph): graph built by Networkx.

    Returns:
        scipy.sparse.matrix: sparse matrix of node-edge relation.
    """
    return _incidence_matrix(graph, weight=None, dtype=np.float32)

def subgraph_construct(graph, graph_list_feasible, graph_list_infeasible):
    """Construct matrix of subgraph.

    Args:
        graph (Networkx.Graph): complete graph built by Networkx
        graph_list_feasible (List[List[str]]): list of feasible subgraph. each subgraph is represented by a list of included nodes.
        graph_list_infeasible (List[List[str]]): list of infeasible subgraph. each subgraph is represented by a list of included nodes.
    """
    nodelist = list(graph.nodes)
    num_nodes = len(nodelist)
    num_subgraphs_feasible = len(graph_list_feasible)
    num_subgraphs_infeasible = len(graph_list_infeasible)

    node_to_index = {node: i for i, node in enumerate(nodelist)}

    # Prepare row, col, and data for COO format
    rows = []
    cols = []
    data = []
    feasibility = []
    row_index = 0

    # Iterate over each subgraph and mark presence of nodes
    for _, subgraph in enumerate(graph_list_feasible):
        if len(subgraph)>=3:
            for node in subgraph:
                rows.append(row_index)            # Subgraph index as row
                cols.append(node_to_index[node])      # Node index as column
                data.append(1)
            row_index += 1                        # Presence indicated by 1
            feasibility.append(1)
    for _, subgraph in enumerate(graph_list_infeasible):
        if len(subgraph)>=3:
            for node in subgraph:
                rows.append(row_index)            # Subgraph index as row
                cols.append(node_to_index[node])      # Node index as column
                data.append(1)
            row_index += 1                        # Presence indicated by 1
            feasibility.append(0)

    return rows, cols, data, feasibility, (row_index, num_nodes)

def prepare_mean_field(data):
    """prepare matrixes for mean field embedding.

    Args:
        data: Dataclass include graph, feasible, and infeasible trips

    Returns:
        n2n matrix, e2n matrix, subgraph matrix (all sparse version)
    """
    graph = data.graph
    graph_list_feasible = data.feasible
    graph_list_infeasible = data.infeasible

    # Create a PyTorch sparse COO tensor for n2n
    n2n_matrix = n2n_construct(graph)
    n2n_indices = torch.tensor(np.array([n2n_matrix.row, n2n_matrix.col]), dtype=torch.long)
    n2n_values = torch.tensor(n2n_matrix.data, dtype=torch.float32)
    shape = n2n_matrix.shape
    n2n_sp = torch.sparse_coo_tensor(n2n_indices, n2n_values, torch.Size(shape))

    # Create a PyTorch sparse COO tensor for e2n
    e2n_matrix = e2n_construct(graph)
    e2n_indices = torch.tensor(np.array([e2n_matrix.row, e2n_matrix.col]), dtype=torch.long)
    e2n_values = torch.tensor(e2n_matrix.data, dtype=torch.float32)
    shape = e2n_matrix.shape
    e2n_sp = torch.sparse_coo_tensor(e2n_indices, e2n_values, torch.Size(shape))

    # Creat a PyTorch sparse COO tensor for subgraph
    subg_sp_rows, subg_sp_cols, subg_sp_data, subg_feasibility, (num_subgraphs, num_nodes) = subgraph_construct(graph, graph_list_feasible, graph_list_infeasible)
    subg_indices = torch.tensor([subg_sp_rows, subg_sp_cols], dtype=torch.long)
    subg_values = torch.tensor(subg_sp_data, dtype=torch.float32)
    subg_shape = (num_subgraphs, num_nodes)
    subg_sp = torch.sparse_coo_tensor(subg_indices, subg_values, torch.Size(subg_shape))   
    subg_feasibility = torch.tensor(subg_feasibility, dtype=torch.float32)

    # Extract node features
    node_feats = []
    for node in graph.nodes(data=True):
        if node[0][0] == 'r':
            node_feats.append(node[1]['wait']/60)
        else:
            node_feats.append(node[1]['onboard'])
    node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(1)

    # Extract edge features
    edge_feats = []
    for edge in graph.edges(data=True):
        edge_feats.append(edge[2]['weight'])
    edge_feats = torch.tensor(edge_feats, dtype=torch.float32).unsqueeze(1)

    # Use StandardScaler for normalization
    scaler_node = StandardScaler()
    scaler_edge = StandardScaler()

    # Fit and transform the node features
    normalized_node_feats = torch.tensor(scaler_node.fit_transform(node_feats), dtype=torch.float32)

    # Fit and transform the edge features
    normalized_edge_feats = torch.tensor(scaler_edge.fit_transform(edge_feats), dtype=torch.float32)

    return n2n_sp, e2n_sp, subg_sp, subg_feasibility, normalized_node_feats, normalized_edge_feats

# Function to evaluate the model
def evaluate_model(model, data_loader, loss_fn, threshold):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    feasible_sample = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            for nen_sp, e2n_sp, subg_sp, subg_feasibility, node_feats, edge_feats in zip(*batch):
                # Forward pass
                outputs = model(nen_sp, e2n_sp, subg_sp, node_feats, edge_feats)

                # Compute loss
                loss = loss_fn(outputs.squeeze(), subg_feasibility)
                total_loss += loss.item()

                # Compute accuracy
                predictions = torch.sigmoid(outputs) > threshold  # Threshold
                correct_preds += (predictions.squeeze() == subg_feasibility).sum().item()
                total_samples += len(subg_feasibility)
                feasible_sample += sum(subg_feasibility)

    accuracy = correct_preds / total_samples
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, feasible_sample/total_samples

# Evaluation function
def evaluate_model_subgraph(model, loader, loss_fn, threshold, device):
    model.eval()
    loss_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch).squeeze()

            loss_total += loss_fn(outputs.squeeze(), batch.y)
            preds = (outputs > threshold).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    loss = loss_total/len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return loss, accuracy