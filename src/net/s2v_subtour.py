import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum,scatter_mean,scatter_add
from src.utils.pytorch_util import weights_init

class Struc2Vec(nn.Module):

    node_type_dict = {
        0: "vehicle",
        1: "pickup",
        2: "dropoff"
    }

    def __init__(self, p_dim, nfeatures_vehicle=2, nfeatures_pickup=3, nfeatures_dropoff=2, nfeatures_edge=1, r=4):
        super(Struc2Vec, self).__init__()

        self.nfeatures_edge = nfeatures_edge

        # Universe layers for different node types
        self.theta1_linear = nn.Linear(p_dim, p_dim) # Node embeddings weights
        self.theta2_linear = nn.Linear(p_dim, p_dim) # Edge aggregation weights
        self.theta3_linear = nn.Linear(nfeatures_edge, p_dim) # Edge embeddings weights

        # Attention mechanism layers
        self.attn_linear = nn.Linear(2 * p_dim + p_dim, 1)  # Attention weights

        # Type-specific layers
        self.type_transform = nn.ModuleDict({
            "0": nn.Linear(nfeatures_vehicle, p_dim), # vehicle node
            "1": nn.Linear(nfeatures_pickup, p_dim), # pickup node
            "2": nn.Linear(nfeatures_dropoff, p_dim), # dropoff node
        })

        # Batch normalization
        # self.batch_norm = nn.ModuleDict({
        #     "0": nn.BatchNorm1d(p_dim),
        #     "1": nn.BatchNorm1d(p_dim),
        #     "2": nn.BatchNorm1d(p_dim),
        # })
        # self.edge_norm = nn.BatchNorm1d(1)
        # self.batch_norm = nn.BatchNorm1d(p_dim)

        # Round of iterations
        self.r = r

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(p_dim, p_dim),
            # nn.Dropout(p=0.5),
            nn.Linear(p_dim, 1),  # Reduce graph embedding to a single value
            nn.Sigmoid()          # Map output to [0, 1] for binary classification
        )
        self.apply(weights_init)

    def forward(self, x_vehicle, x_pickup, x_dropoff, edge_index, edge_attr, node_types, mu, batch):
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

        # r round of iterations
        # src, dst = edge_index[0], edge_index[1]  # Source and target nodes
        for _ in range(self.r):
            ti = edge_attr
            transformed_ti = F.leaky_relu(
                self.theta3_linear(ti.view(-1, self.nfeatures_edge))
            )

            # # Attention mechanism
            concatenated_features = torch.cat([mu[edge_index[0]], mu[edge_index[1]], transformed_ti], dim=1)  # Concatenate features of source, target, edge
            attention_scores = self.attn_linear(concatenated_features)  # Compute attention scores
            # attention_scores = scatter_softmax(attention_scores, edge_index[1], dim=0)  # Normalize scores per destination
            exp_scores = torch.exp(attention_scores)
            sum_exp = scatter_add(exp_scores, edge_index[1], dim=0, dim_size=node_types.size(0))
            attention_scores = exp_scores / sum_exp[edge_index[1]]  # Normalize scores per destination

            # Message passing with attention
            weighted_messages = attention_scores * mu[edge_index[0]]  # Scale messages by attention
            aggregated_mu = scatter_sum(weighted_messages, edge_index[1], dim=0, dim_size=node_types.size(0))

            weighted_edges = attention_scores * transformed_ti
            aggregated_ti = scatter_sum(weighted_edges, edge_index[1], dim=0, dim_size=node_types.size(0))

            # Message passing without attention
            # aggregated_mu = scatter_sum(mu[edge_index[0]],
            #                             edge_index[1],
            #                             dim=0,
            #                             dim_size=node_types.size(0))
            # aggregated_ti = scatter_sum(
            #     transformed_ti,
            #     edge_index[1],
            #     dim=0,
            #     dim_size=node_types.size(0))
            

            # Message passing
            mu = self.theta1_linear(aggregated_mu) + self.theta2_linear(aggregated_ti)

            # Node type-specific transformations
            for node_type in self.type_transform:
                mask = node_types == int(node_type)
                if node_type == "0":
                    mu[mask] = mu[mask] + self.type_transform[node_type](x_vehicle)
                elif node_type == "1":
                    mu[mask] = mu[mask] + self.type_transform[node_type](x_pickup)
                elif node_type == "2":
                    mu[mask] = mu[mask] + self.type_transform[node_type](x_dropoff)
                else:
                    raise ValueError(f"Invalid node type: {node_type}")

            mu = F.leaky_relu(mu)
        
        # Graph aggregation
        graph_embeddings = scatter_mean(mu, batch, dim=0)  # (num_graphs, p_dim)

        # Classification
        proba = self.classifier(graph_embeddings)

        return proba