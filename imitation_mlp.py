import pickle
import wandb
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Precision, Recall
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
from typing import List
import networkx as nx
from src.net.s2v import MLP
from src.utils.pytorch_util import evaluate_model_subgraph
from src.utils.helper import DataPoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


epochs = 20
threshold = 0.5
class SubgraphDataset(Dataset):
    def __init__(self, data_points):
        self.samples = []
        
        for data_point in data_points:
            # Process feasible subgraphs
            for nodes in data_point.feasible:
                if len(nodes) == 4:
                    if data_point.graph.subgraph(nodes).number_of_edges() + len(nodes)!=10:
                        break
                    self.samples.append((self.flatten_subgraph(data_point.graph, nodes), 1))
            
            # Process infeasible subgraphs
            for nodes in data_point.infeasible:
                if len(nodes) == 4:
                    self.samples.append((self.flatten_subgraph(data_point.graph, nodes), 0))
    
    def flatten_subgraph(self, graph, nodes):
        """Flatten node and edge features of a subgraph."""
        subgraph = graph.subgraph(nodes)
        sorted_nodes = sorted(subgraph.nodes, key=lambda n: (str(n).startswith('r'), n))

        # Flatten node features
        node_features = [
            subgraph.nodes[n]['onboard'] if str(n).startswith('v') else subgraph.nodes[n]['wait'] / 60
            for n in sorted_nodes
        ]
        
        node_index_map = {node: idx for idx, node in enumerate(sorted_nodes)}

        # Flatten edge features
        sorted_edges = sorted(
            subgraph.edges,
            key=lambda e: (node_index_map[e[0]], node_index_map[e[1]])
        )
        edge_features = [graph.edges[e]['weight'] for e in sorted_edges]
        
        # Concatenate and return flattened feature vector
        flattened_features = torch.tensor(node_features + edge_features, dtype=torch.float)

        if flattened_features.shape[0]!=10:
            print(flattened_features.shape)

        return flattened_features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return features, torch.tensor(label, dtype=torch.float32)
    
filepath = './data/imitation/data_points_connected_clean.pkl'
with open(filepath, 'rb') as f:
    data_points = pickle.load(f)
print(f"Data points loaded from {filepath}")
dataset = SubgraphDataset(data_points)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Set up training model ...")
model = MLP(input_size=10, 
            hidden_size=64,
            output_size=1)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

precision = Precision(task="binary")  # Micro for binary
recall = Recall(task="binary")

print("Training start...")

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        # batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch[0]).squeeze()

        # print("Logits:", F.sigmoid(sampled_outputs).squeeze()[:10])

        # Compute loss
        loss = loss_fn(outputs, batch[1])

        # Backward pass and optimizer step
        loss.backward()

        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm()}")

        # for name, param in model.named_parameters():
        #     if "weight" in name:
        #         print(f"Layer: {name}, Weights: {param.data}")
        #     if "bias" in name:
        #         print(f"Layer: {name}, Bias: {param.data}")

        # Track total loss and accuracy
        total_loss += loss.item()

        preds = (outputs > threshold).float()  # Convert logits to binary predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch[1].cpu().numpy())

        precision.update(preds, batch[1])
        recall.update(preds, batch[1])

    # Calculate epoch accuracy
    train_accuracy = accuracy_score(all_labels, all_preds)


    print(f"Epoch [{epoch+1}/{epochs}], ", f"Train Loss: {total_loss/len(dataloader):.4f}, Train Accuracy: {train_accuracy:.4f}",
        f"Precision: {precision.compute():.4f}", f"Recall: {recall.compute():.4f}")
    precision.reset()
    recall.reset()
