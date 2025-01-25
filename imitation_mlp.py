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
from src.utils.pytorch_util import evaluate_model_mlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


epochs = 1000
lr = 1e-3
threshold = 0.5
train_split = 0.8
batch_size = 64
WANDB = 1
VERBOSE = 0
size = 4
num_features = size + size*(size-1)/2
class SubgraphDataset(Dataset):
    def __init__(self, data_points):
        self.samples = []
        
        for data_point in data_points:
            # Make balanced dataset
            feasible_num = 0
            infeasible_num = 0
            for nodes in data_point.feasible:
                if len(nodes) == size:
                    feasible_num += 1
            for nodes in data_point.infeasible:
                if len(nodes) == size:
                    infeasible_num += 1        
            num = min(feasible_num,infeasible_num)
            count = 0
            # Process feasible subgraphs
            for nodes in data_point.feasible:
                if len(nodes) == size:
                    if data_point.graph.subgraph(nodes).number_of_edges() + len(nodes)!=num_features:
                        raise ValueError("Feasible trip not connected!")
                    self.samples.append((self.flatten_subgraph(data_point.graph, nodes), 1))
                    count += 1
                    if count >= num:
                        break
            
            count = 0
            # Process infeasible subgraphs
            for nodes in data_point.infeasible:
                if len(nodes) == size:
                    if data_point.graph.subgraph(nodes).number_of_edges() + len(nodes)!=num_features:
                        raise ValueError("Infeasible trip not connected!")
                    self.samples.append((self.flatten_subgraph(data_point.graph, nodes), 0))
                    count += 1
                    if count >= num:
                        break
    
    def flatten_subgraph(self, graph, nodes):
        """Flatten node and edge features of a subgraph."""
        subgraph = graph.subgraph(nodes)

        # Order nodes
        v_node = [n for n in subgraph.nodes if n.startswith('v')][0]
        rv_weights = {node:subgraph.get_edge_data(v_node,node)['weight'] for node in subgraph.nodes if node.startswith('r')}
        rv_weights[v_node] = 0
        sorted_nodes = sorted(subgraph.nodes, key=lambda n: (n.startswith('v'), rv_weights[n]))

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

        if flattened_features.shape[0]!= num_features:
            print(flattened_features.shape)

        return flattened_features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return features, torch.tensor(label, dtype=torch.float32)
    
filepath = './data/imitation/data_points_connected_delay.pkl'
with open(filepath, 'rb') as f:
    data_points = pickle.load(f)
print(f"Data points loaded from {filepath}")
dataset = SubgraphDataset(data_points[10:20])
positive = sum([1 for each in dataset if each[1] == 0])
print(f"Positive/negative ratio:{positive/(len(dataset)-positive):.2f}")
print(f"Dataset size: {len(dataset)}")
train_data, test_data = train_test_split(dataset, test_size=1-train_split, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=0)

print("Set up training model ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_size=int(num_features),
            hidden_size=64,
            output_size=1).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

precision = Precision(task="binary")  # Micro for binary
recall = Recall(task="binary")

print("Training start...")
if WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RV-imitation",

        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "MLP",
        "dataset": "size 3",
        "batch_size": batch_size,
        }
    )

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        features = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features).squeeze()

        # print("Logits:", F.sigmoid(sampled_outputs).squeeze()[:10])

        # Compute loss
        loss = loss_fn(outputs, labels)

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
        all_labels.extend(labels.cpu().numpy())

        precision.update(preds, labels)
        recall.update(preds, labels)

    # Calculate epoch accuracy
    train_accuracy = accuracy_score(all_labels, all_preds)
    test_loss, test_accuracy = evaluate_model_mlp(model, test_loader, loss_fn, threshold, device)

    if WANDB:
        wandb.log({"Train Loss": total_loss/len(train_loader), "Train Accuracy": train_accuracy,
            "Test Loss": test_loss, "Test Accuracy": test_accuracy,
            "Precision": precision.compute(), "Recall": recall.compute()})

    if VERBOSE:
        print(f"Epoch [{epoch+1}/{epochs}], ", f"Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}",
            f"Precision: {precision.compute():.4f}", f"Recall: {recall.compute():.4f}", f"Test Loss: {test_loss}", f"Test Accuracy: {test_accuracy}")
    precision.reset()
    recall.reset()
