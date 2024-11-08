import os
import sys
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
        
        self.conv = nn.Linear(latent_dim, latent_dim)
        # Weight initialization
        self.apply(weights_init)

    def foward(self, graph, graph_list, node_feats, edge_feats):
        n2n_sp, e2n_sp, subg_sp = prepare_mean_field(graph, graph_list)

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
        input_potential = F.relu(input_message)

        # Multistep update
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = torch.sparse.mm(n2n_sp, cur_message_layer)
            node_linear = self.conv(n2npool)
            merged_linear = node_linear + input_message

            cur_message_layer = F.relu(merged_linear)
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer
            
        y_potential = torch.sparse.mm(subg_sp, reluact_fp)

        return F.relu(y_potential)  
    
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