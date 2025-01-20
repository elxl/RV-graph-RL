import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum,scatter_mean
from src.utils.pytorch_util import weights_init

class Struc2Vec(nn.Module):

    node_type_dict = {
        0: "vehicle",
        1: "pickup",
        2: "dropoff"
    }

    def __init__(self, p_dim, nfeatures_vehicle=2, nfeatures_pickup=3, nfeatures_dropoff=2, r=4):
        super(Struc2Vec, self).__init__()

        # Universe layers for different node types
        self.theta1_linear = nn.Linear(p_dim, p_dim) # Node embeddings weights
        self.theta2_linear = nn.Linear(p_dim, p_dim) # Edge embeddings weights
        self.theta3_linear = nn.Linear(1, p_dim) # Node embeddings

        # Type-specific layers
        self.type_transform = nn.ModuleDict({
            "0": nn.Linear(nfeatures_vehicle, p_dim), # vehicle node
            "1": nn.Linear(nfeatures_pickup, p_dim), # pickup node
            "2": nn.Linear(nfeatures_dropoff, p_dim), # dropoff node
        })

        # Round of iterations
        self.r = r

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(p_dim, 1),  # Reduce graph embedding to a single value
            nn.Sigmoid()          # Map output to [0, 1] for binary classification
        )
        self.apply(weights_init)

    def forward(self, data):
        """
         Args:
            data: torch_geometric.data.Batch object containing:
                - x_vehicle: Features for vehicle nodes (V, 2)
                - x_pickup: Features for pickup nodes (P, 3)
                - x_dropoff: Features for dropoff nodes (D, 2)
                - edge_index: Edge list (2, E)
                - edge_attr: Edge features (E, 1)
                - node_types: Node type mapping (N,)
                - node_mu: Initial values of mu (N, p_dim)
        """
        edge_index, edge_attr, node_types, batch = (
            data.edge_index,
            data.edge_attr,
            data.node_types,
            data.batch,
        )        
        mu = data.node_mu

        # r round of iterations
        for _ in range(self.r):
            ti = edge_attr
            transformed_ti = F.leaky_relu(
                self.theta3_linear(ti.view(-1, 1))
            )
            aggregated_ti = scatter_sum(
                transformed_ti,
                edge_index[1],
                dim=0,
                dim_size=node_types.size(0)
            )
            aggregated_mu = scatter_sum(mu[edge_index[0]],
                                        edge_index[1],
                                        dim=0,
                                        dim_size=node_types.size(0))
            # Message passing
            mu = self.theta1_linear(aggregated_mu) + self.theta2_linear(aggregated_ti)

            # Node type-specific transformations
            for node_type in self.type_transform:
                mask = node_types == int(node_type)
                mu[mask] = mu[mask] + self.type_transform[node_type](data[f"x_{self.node_type_dict[int(node_type)]}"])

            mu = F.leaky_relu(mu)
        
        # Graph aggregation
        graph_embeddings = scatter_mean(mu, batch, dim=0)  # (num_graphs, p_dim)

        # Classification
        proba = self.classifier(graph_embeddings)

        return proba