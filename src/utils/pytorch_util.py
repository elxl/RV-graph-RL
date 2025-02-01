import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import networkx as nx
import scipy.sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data

def prepare_graph(timestep, vehicle, requests, network, directed=False, evaluation=False):
    """Preapre a graph data point for a single trip

    Args:
        timestep (int)
        vehicle (Vehicle)
        requests (List[Request])
        network (network)
        directed (bool)

    Returns:
        torch_geometric.data.Data
    """
    node_feats_vehicle = []
    node_feats_pickup = []
    node_feats_dropoff = []
    deadlines = [] # deadline for visting node
    edge_index = []
    edge_feats = []
    node_types = []
    node_number = [] # map from index to node
    node_idx = 0

    # Vehicle node
    node_feats_vehicle.append([vehicle.capacity, len(vehicle.passengers)])
    node_idx += 1
    node_types.append(0) # 0:vehicle, 1:pickup, 2:dropoff
    node_number.append(vehicle.node)
    deadlines.append(np.inf)

    # New request origin nodes
    for request in requests:
        origin_idx = node_idx

        # node_feats_pickup.append([(request.latest_boarding - timestep)/60, 1, network.get_time(vehicle.node, request.origin)/60])
        deadline_residual = (request.latest_boarding - timestep)/60
        node_feats_pickup.append([deadline_residual, 1])
        node_idx += 1
        node_types.append(1)
        node_number.append(request.origin)
        deadlines.append(deadline_residual)

        # Connect vehicle node to origin node
        edge_index.append([0, origin_idx])
        od_travel = network.get_time(vehicle.node, request.origin)/60
        edge_feats.append([od_travel, deadline_residual-od_travel])
        if not directed:
            edge_index.append([origin_idx, 0])
            od_travel = network.get_time(request.origin, vehicle.node)/60
            edge_feats.append([od_travel, deadlines[0]-od_travel])
        
        # Connect to other origin nodes
        for node in range(1,node_idx-1):
            edge_index.append([node, origin_idx])
            edge_index.append([origin_idx, node])
            od_travel_1 = network.get_time(node_number[node], request.origin)/60
            od_travel_2 = network.get_time(request.origin, node_number[node])/60
            edge_feats.append([od_travel_1, deadline_residual-od_travel_1])
            edge_feats.append([od_travel_2, deadlines[node]-od_travel_2])

    # New request destination nodes
    for request in requests:
        destination_idx = node_idx

        deadline_residual = (request.latest_alighting - timestep)/60
        node_feats_dropoff.append([deadline_residual, -1])
        node_idx += 1
        node_types.append(2)
        node_number.append(request.destination)
        deadlines.append(deadline_residual)

        # Connect to other nodes (except vehicle node)
        for node in range(1,node_idx-1):
            edge_index.append([node, destination_idx])
            od_travel_1 = network.get_time(node_number[node], request.destination)/60
            edge_feats.append([od_travel_1, deadline_residual-od_travel_1])
            if directed and node_number[node] == request.origin:
                continue
            edge_index.append([destination_idx, node])
            od_travel_2 = network.get_time(request.destination, node_number[node])/60
            edge_feats.append([od_travel_2, deadlines[node]-od_travel_2])

    # Onboard destination node
    for request in vehicle.passengers:
        destination_idx = node_idx

        deadline_residual = (request.latest_alighting - timestep)/60
        node_feats_dropoff.append([deadline_residual, -1])
        node_idx += 1
        node_types.append(2)
        node_number.append(request.destination)
        deadlines.append(deadline_residual)

        # Connect vehicle node to destination node
        edge_index.append([0, destination_idx])
        od_travel = network.get_time(vehicle.node, request.destination)/60
        edge_feats.append([od_travel, deadline_residual-od_travel])
        if not directed:
            edge_index.append([destination_idx, 0])
            od_travel = network.get_time(request.destination, vehicle.node)/60
            edge_feats.append([od_travel, deadlines[0]-od_travel])

        # Connect to other nodes (except vehicle node)
        for node in range(1,node_idx-1):
            edge_index.append([node, destination_idx])
            edge_index.append([destination_idx, node])
            od_travel_1 = network.get_time(node_number[node], request.destination)/60
            od_travel_2 = network.get_time(request.destination, node_number[node])/60
            edge_feats.append([od_travel_1, deadline_residual-od_travel_1])
            edge_feats.append([od_travel_2, deadlines[node]-od_travel_2])

    if not evaluation:
        # Convert to tensor
        node_feats_vehicle = torch.tensor(node_feats_vehicle, dtype=torch.float)
        node_feats_pickup = torch.tensor(node_feats_pickup, dtype=torch.float)
        node_feats_dropoff = torch.tensor(node_feats_dropoff, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)
        node_types = torch.tensor(node_types, dtype=torch.long)
        node_number = torch.tensor(node_number, dtype=torch.long)

        graph = Data(x_vehicle=node_feats_vehicle, 
                    x_pickup=node_feats_pickup,
                    x_dropoff=node_feats_dropoff,
                    edge_index=edge_index, 
                    edge_attr=edge_feats, 
                    node_types=node_types,
                    node_number=node_number)
        return graph
    else:
        node_feats_vehicle = np.array(node_feats_vehicle, dtype=np.float32)
        node_feats_pickup = np.array(node_feats_pickup, dtype=np.float32)
        node_feats_dropoff = np.array(node_feats_dropoff, dtype=np.float32)
        edge_index = np.array(edge_index, dtype=np.int64)
        edge_feats = np.array(edge_feats, dtype=np.float32)
        node_types = np.array(node_types, dtype=np.int64)

        node_feats_vehicle = torch.from_numpy(node_feats_vehicle)
        node_feats_pickup = torch.from_numpy(node_feats_pickup)
        node_feats_dropoff = torch.from_numpy(node_feats_dropoff)
        edge_index = torch.from_numpy(edge_index).t().contiguous()
        edge_feats = torch.from_numpy(edge_feats)
        node_types = torch.from_numpy(node_types)
        return node_feats_vehicle, node_feats_pickup, node_feats_dropoff, edge_index, edge_feats, node_types

def process_trip_lists(timestep, trip, network, label=1, directed=True, evaluation=False):
    """Process a list of trip lists into a list of graph data points

    Args:
        timestep (int)
        trip (List[Vehicle, List[Request]])
        network (Network)
        label (int)
        directed (bool)

    Returns:
        List[torch_geometric.data.Data]
    """
    vehicle, requests = trip[0], trip[1:]
    if not evaluation:
        graph = prepare_graph(timestep, vehicle, requests, network, directed, evaluation)
        graph.y = torch.tensor([label], dtype=torch.long)
        return graph
    else:
        return prepare_graph(timestep, vehicle, requests, network, directed, evaluation)

def add_feature_initialization(data_point, p_dim):
    """Add initial features for each data point

    Args:
        data_point (List[torch_geometric.data.Data])
        p_dim (int)

    Returns:
        List[torch_geometric.data.Data]
    """
    num_nodes = data_point.x_vehicle.size(0) + data_point.x_pickup.size(0) + data_point.x_dropoff.size(0)

    # Initialize a zero vector for each node
    node_mu = torch.zeros(num_nodes, p_dim)

    # Add the new attribute to the data point
    data_point.node_mu = node_mu
    return data_point

def convert2onxx(model, checkpoint_path, onnx_path, example_input = "data/example_input.pt"):
    """Convert PyTorch model to ONNX format"""
    # Load the best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare dummy input
    data_point = torch.load(example_input)
    print(f"Example input loaded from {example_input}.")
    batch = torch.zeros(data_point.node_types.size(0), dtype=torch.long, device=data_point.node_types.device)
    example_input = (
        data_point.x_vehicle, 
        data_point.x_pickup,
        data_point.x_dropoff, 
        data_point.edge_index,
        data_point.edge_attr,
        data_point.node_types,
        data_point.node_mu, 
        batch,
    )
    # Convert the model to ONNX format
    torch.onnx.export(
        model,                                  # PyTorch model
        example_input,                          # Example input tensor
        onnx_path,                              # Output file name
        export_params=True,                     # Store trained weights
        opset_version=11,                       # ONNX opset version (11+ recommended)
        do_constant_folding=True,               # Optimize constant folding
        input_names=["x_vehicle","x_pickup","x_dropoff","edge_index","edge_attr","node_types","mu","batch"],                  # Name of the input layer(s)
        output_names=["output"],                # Name of the output layer(s)
        dynamic_axes={                          # Dynamic axes for variable input sizes
            "x_vehicle": {0: "num_vehicle_nodes"},   # Allow variable vehicle nodes
            "x_pickup": {0: "num_pickup_nodes"},     # Allow variable pickup nodes
            "x_dropoff": {0: "num_dropoff_nodes"},   # Allow variable dropoff nodes
            "edge_index": {1: "num_edges"},          # Allow variable edges
            "edge_attr": {0: "num_edges"},           # Allow variable edge features
            "node_types": {0: "num_nodes"},          # Allow variable node types
            "mu": {0: "num_nodes"},                  # Allow variable node embeddings
            "batch": {0: "num_nodes"}                # Allow variable batch sizes
        })
    print(f"Pytorch model {checkpoint_path} exported to ONNX format {onnx_path}.")

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

def evaluate_model_mlp(model, loader, loss_fn, threshold, device):
    model.eval()
    loss_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(features).squeeze()
            loss_total += loss_fn(outputs, labels)

            preds = (outputs > threshold).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    loss = loss_total/len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return loss, accuracy

def evaluate_model_s2v(model, loader, loss_fn, threshold, device, mislabled=False):
    model.eval()
    loss_total = 0
    total_data_points = 0

    all_preds = []
    all_labels = []
    mislabeled_labels = []
    mislabeled_scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_vehicle, x_pickup, x_dropoff, edge_index, edge_attr, node_types, batch_index = (
                batch.x_vehicle,
                batch.x_pickup,
                batch.x_dropoff,
                batch.edge_index,
                batch.edge_attr,
                batch.node_types,
                batch.batch,
            )
            mu = batch.node_mu

            # Forward pass
            outputs = model(x_vehicle, x_pickup, x_dropoff, edge_index, edge_attr, node_types, mu, batch_index)

            loss_total += loss_fn(outputs, batch.y.view(-1, 1).float()) * batch.y.size(0)
            total_data_points += batch.y.size(0)
            preds = (outputs > threshold).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            if mislabled:
                for i, pred in enumerate(preds):
                    if pred.item() != batch.y[i].item():
                        mislabeled_labels.append(batch.y[i].cpu().item())
                        mislabeled_scores.append(outputs[i].cpu().item())

    loss = loss_total / total_data_points
    accuracy = accuracy_score(all_labels, all_preds)
    return loss, accuracy, mislabeled_labels, mislabeled_scores