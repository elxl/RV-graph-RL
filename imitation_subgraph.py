import pickle
import wandb
import random
from dataclasses import dataclass
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
from src.net.s2v import GraphClassifier
from src.utils.pytorch_util import evaluate_model_subgraph
from src.utils.helper import DataPoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="Imitation learning.")
parser.add_argument(
    '--DATA', 
    type=str,
    default='./data/imitation/data_points_connected.pkl',
    help="Root directory for data.")
parser.add_argument(
    '--LATENT_DIMENSION',
    type=int,
    default=64,
    help="Dimension of latent features."
)
parser.add_argument(
    '--NODE_FEATURE',
    type=int,
    default=1,
    help="Number of node features."
)
parser.add_argument(
    '--EDGE_FEATURE',
    type=int,
    default=1,
    help="Number of edge features."
)
parser.add_argument(
    '--NUM_EMBEDDING',
    type=int,
    default=3,
    help="Number of iterations for embedding."
)
parser.add_argument(
    '--EPOCH',
    type=int,
    default=20,
    help="Number of training epochs."
)
parser.add_argument(
    '--THRESHOLD',
    type=float,
    default=0.5,
    help="Threshold to classify trip as feasible."
)
parser.add_argument(
    '--TRAIN_SPLIT',
    type=float,
    default=0.8,
    help="Train-test split."
)
parser.add_argument(
    '--BATCH',
    type=int,
    default=32,
    help="Batch size for training."
)
parser.add_argument(
    '--LR',
    type=float,
    default=1e-4,
    help="Learning rate."
)
parser.add_argument(
    '--NORM',
    type=int,
    default=0,
    help="Feature normalization."
)
parser.add_argument(
    '--WANDB',
    type=int,
    default=0,
    help="Either using wandb or not."
)
parser.add_argument(
    '--VERBOSE',
    type=int,
    default=0,
    help="Either output training statistic to terminal."
)

def load_data_points(filepath):
    with open(filepath, 'rb') as f:
        data_points = pickle.load(f)
    print(f"Data points loaded from {filepath}")
    return data_points

def normalize_tensor(tensor, dim=0):
    """Normalize a tensor along the specified dimension."""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    std[std == 0] = 1  # Avoid division by zero
    return (tensor - mean) / std

def extract_subgraph(graph, nodes):
    """Extract subgraph and convert to PyTorch Geometric Data format."""
    subgraph = graph.subgraph(nodes)
    node_mapping = {node: i for i, node in enumerate(subgraph.nodes)}

    edge_index = torch.tensor(
        [[node_mapping[u], node_mapping[v]] for u, v in subgraph.edges], dtype=torch.long
    ).t().contiguous()

    # Extract node features
    node_features = []
    for n in subgraph.nodes:
        if str(n).startswith('v'):
            node_features.append(subgraph.nodes[n]['onboard'])
        elif str(n).startswith('r'):
            node_features.append(subgraph.nodes[n]['wait']/60)
        else:
            raise ValueError(f"Unknown node type for node {n}")
        
    node_features = torch.tensor(node_features, dtype=torch.float).unsqueeze(dim=1)  # Ensure Tensor
    
    # Extract edge features
    edge_features = [subgraph[u][v].get('weight') for u, v in subgraph.edges]
    edge_features = torch.tensor(edge_features, dtype=torch.float).unsqueeze(dim=1)

    if args.NORM:
        node_features = normalize_tensor(node_features, dim=0)  # Normalize node features
        edge_features = normalize_tensor(edge_features, dim=0)  # Normalize edge features

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

def preprocess_data(data_points: List[DataPoint]):
    """Preprocess data points into PyTorch Geometric Data format."""
    dataset = []
    
    for data in data_points:
        #Process feasible subgraphs with label 1
        num_sample = min(len(data.feasible),len(data.infeasible))
        batch_int = (num_sample//args.BATCH)
        num_sample = batch_int * args.BATCH
        num = 0
        for nodes in data.feasible:
            if len(nodes)>=3:
                subgraph_data = extract_subgraph(data.graph, nodes)
                subgraph_data.y = torch.tensor([1], dtype=torch.float)
                dataset.append(subgraph_data)
                num +=1
                if num>=num_sample:
                    break
        
        # Process infeasible subgraphs with label 0
        num=0
        for nodes in data.infeasible:
            if len(nodes)>=3:
                subgraph_data = extract_subgraph(data.graph, nodes)
                subgraph_data.y = torch.tensor([0], dtype=torch.float)
                dataset.append(subgraph_data)
                num+=1
                if num>=num_sample:
                    break
    
    return dataset
    
args = parser.parse_args()
epochs = args.EPOCH
# Load the data points
print("Load data...")
if args.DATA.endswith("pkl"):
    data_points = load_data_points(args.DATA)

    # Use the DataLoader for batch processing
    dataset = preprocess_data(data_points[10:20])
    # filepath = args.DATA.split(".pkl")[0] + '.pt'
    # torch.save(dataset, filepath)
    # print(f"Data processed. Processed file saved to {filepath}")
else:
    dataset = torch.load(args.DATA)
    print(f"Data loaded from {args.DATA}")
# Split the dataset into train and test sets (80-20 split)
dataset_sampled = dataset#random.sample(dataset, 64000)
positive = 0
negative = 0
for each in dataset_sampled:
    if each.y[0] == 0:
        negative += 1
    else:
        positive += 1
print(f"Sample ratio {(positive/negative):.2f}")
train_data, test_data = train_test_split(dataset_sampled, test_size=1-args.TRAIN_SPLIT, random_state=42)

train_loader = DataLoader(train_data, batch_size=args.BATCH, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data, batch_size=args.BATCH, shuffle=False,num_workers=0)

print("Set up training model ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Binary Cross-Entropy with Logits
loss_fn = nn.BCELoss()

# Initialize the model, optimizer
model = GraphClassifier(
    node_in_channels=1,
    edge_in_channels=1,
    hidden_channels=args.LATENT_DIMENSION,
    out_channels=1,
    max_lv=3
)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.LR)

# Training
print("Training start...")

# start a new wandb run to track this script
if args.WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RV-imitation",

        # track hyperparameters and run metadata
        config={
        "latent_dimension": args.LATENT_DIMENSION,
        "learning_rate": args.LR,
        "architecture": "graph_message_pass",
        "dataset": "data_tiny",
        "epochs": args.EPOCH,
        }
    )

precision = Precision(task="binary")  # Micro for binary
recall = Recall(task="binary")

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch).squeeze()

        # print("Logits:", F.sigmoid(sampled_outputs).squeeze()[:10])

        # Compute loss
        loss = loss_fn(outputs.squeeze(), batch.y)

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

        preds = (outputs > args.THRESHOLD).float()  # Convert logits to binary predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

        precision.update(preds, batch.y)
        recall.update(preds, batch.y)

    # Calculate epoch accuracy
    train_accuracy = accuracy_score(all_labels, all_preds)

    # Evaluate on test set after each epochloss_fn, 
    test_loss, test_accuracy = evaluate_model_subgraph(model, test_loader, loss_fn, args.THRESHOLD, device)

    if args.WANDB:
        wandb.log({"Train Loss": total_loss/len(train_loader), "Train Accuracy": train_accuracy,
            "Test Loss": test_loss, "Test Accuracy": test_accuracy,
            "Precision": precision.compute(), "Recall": recall.compute()})

    if args.VERBOSE:
        print(f"Epoch [{epoch+1}/{epochs}], ", f"Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}",
            f"Precision: {precision.compute():.4f}", f"Recall: {recall.compute():.4f}",
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    precision.reset()
    recall.reset()

    if epoch%50 == 0:
        torch.save(model.state_dict(),"./results/steps_10_20_all_delay.pt")