import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax

from src.utils.pytorch_util import weights_init, prepare_mean_field

class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3) -> None:
        super(EmbedMeanField, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim) # Node feature to latent feature
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim) # Edge feature to latent feature
        if output_dim >0 :
            self.out = nn.Linear(latent_dim, output_dim) # output layer
        
        self.conv1 = nn.Linear(latent_dim, latent_dim)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        # Weight initialization
        self.apply(weights_init)

    def forward(self, n2n_sp, e2n_sp, subg_sp, node_feats, edge_feats):

        device = node_feats.device
        n2n_sp = n2n_sp.to(device)
        e2n_sp = e2n_sp.to(device)
        subg_sp = subg_sp.to(device)

        h = self.mean_field(node_feats, edge_feats, n2n_sp, e2n_sp, subg_sp)
        return h

    def mean_field(self, node_feats, edge_feats, n2n_sp, e2n_sp, subg_sp):
        # Node feature message
        input_node_linear = self.w_n2l(node_feats)
        input_message = input_node_linear

        # Edge feature message
        if edge_feats is not None:
            input_edge_linear = self.w_e2l(edge_feats)
            e2npool_input = torch.sparse.mm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = self.relu(input_message)

        # Multistep update
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = torch.sparse.mm(n2n_sp, cur_message_layer)
            node_linear = self.conv1(n2npool)
            merged_linear = node_linear + input_message

            cur_message_layer = self.relu(merged_linear)
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out(cur_message_layer)
            reluact_fp = self.relu(out_linear)
        else:
            reluact_fp = cur_message_layer
            
        y_potential = torch.sparse.mm(subg_sp, reluact_fp)

        return self.relu(y_potential)  
    
class EdgeFeatureConv(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')  # or 'mean', 'max'
        self.node_mlp = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Linear(edge_in_channels, out_channels)
        self.update_mlp = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j is the source node feature; edge_attr is the edge feature
        return self.node_mlp(x_j) + self.edge_mlp(edge_attr)

    def update(self, aggr_out):
        # Final node update
        return self.update_mlp(aggr_out)
    
class GraphClassifier(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, max_lv=3):
        super(GraphClassifier, self).__init__()
        self.max_lv = max_lv
        self.w_n2l = nn.Linear(node_in_channels, hidden_channels)
        self.w_e2l = nn.Linear(edge_in_channels, hidden_channels)
        self.conv1 = EdgeFeatureConv(hidden_channels, hidden_channels, hidden_channels)
        self.conv2 = EdgeFeatureConv(hidden_channels, hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial feature transformation
        x = self.w_n2l(x)
        edge_attr = self.w_e2l(edge_attr)

        # Perform message-passing with graph convolutions
        for _ in range(self.max_lv):
            x = self.conv1(x, edge_index, edge_attr)
            x = self.relu(x)
            x = self.conv2(x, edge_index, edge_attr)
            x = self.relu(x)

        # Global pooling
        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        # Graph-level classification
        out = self.mlp(pooled)
        return torch.sigmoid(out)  # Apply sigmoid for binary classification
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden to hidden
        self.fc3 = nn.Linear(hidden_size, output_size) # Hidden to output
    
    def forward(self, x):
        # Forward pass through layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation if regression)
        return torch.sigmoid(x)
    
# class EmbedLoopyBP(nn.Module):
#     def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
#         super(EmbedLoopyBP, self).__init__()
#         self.latent_dim = latent_dim
#         self.output_dim = output_dim
#         self.num_node_feats = num_node_feats
#         self.num_edge_feats = num_edge_feats

#         self.max_lv = max_lv

#         self.w_n2l = nn.Linear(num_node_feats, latent_dim)
#         if num_edge_feats > 0:
#             self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
#         if output_dim > 0:
#             self.out = nn.Linear(latent_dim, output_dim)

#         self.conv = nn.Linear(latent_dim, latent_dim)
#         weights_init(self)

#     def forward(self, graph_list, node_feats, edge_feats): 
#         n2e_sp, e2e_sp, e2n_sp, subg_sp = PrepareLoopyBP(graph_list)

#         device = node_feats.device
#         n2e_sp = n2e_sp.to(device)
#         e2e_sp = e2e_sp.to(device)
#         e2n_sp = e2n_sp.to(device)
#         subg_sp = subg_sp.to(device)

#         h = self.loopy_bp(node_feats, edge_feats, n2e_sp, e2e_sp, e2n_sp, subg_sp)
        
#         return h

#     def loopy_bp(self, node_feats, edge_feats, n2e_sp, e2e_sp, e2n_sp, subg_sp):
#         input_node_linear = self.w_n2l(node_feats)
#         n2epool_input = torch.sparse.mm(n2e_sp, input_node_linear)
#         input_message = n2epool_input

#         if edge_feats is not None:
#             input_edge_linear = self.w_e2l(edge_feats)
#             input_message += input_edge_linear
            
#         input_potential = F.relu(input_message)

#         lv = 0
#         cur_message_layer = input_potential
#         while lv < self.max_lv:
#             e2epool = torch.sparse.mm(e2e_sp, cur_message_layer)
#             edge_linear = self.conv(e2epool)
#             merged_linear = edge_linear + input_message

#             cur_message_layer = F.relu(merged_linear)
#             lv += 1

#         e2npool = torch.sparse.mm(e2n_sp, cur_message_layer)
#         hidden_msg = F.relu(e2npool)
#         if self.output_dim > 0:
#             out_linear = self.out(hidden_msg)
#             reluact_fp = F.relu(out_linear)
#         else:
#             reluact_fp = hidden_msg

#         y_potential = torch.sparse.mm(subg_sp, reluact_fp)

#         return F.relu(y_potential)