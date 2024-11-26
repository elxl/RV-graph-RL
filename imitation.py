import pickle
import wandb
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Precision, Recall
import argparse
from typing import List
import networkx as nx
from src.utils.pytorch_util import prepare_mean_field
from src.net.s2v import EmbedMeanField
from src.utils.pytorch_util import evaluate_model
from src.utils.helper import DataPoint

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
    default=4,
    help="Batch size for training."
)
parser.add_argument(
    '--LR',
    type=float,
    default=1e-5,
    help="Learning rate."
)
parser.add_argument(
    '--WANDB',
    type=int,
    default=0,
    help="Either using wandb or not."
)

def load_data_points(filepath):
    with open(filepath, 'rb') as f:
        data_points = pickle.load(f)
    print(f"Data points loaded from {filepath}")
    return data_points

class GraphDataset(Dataset):
    def __init__(self, data_points, prepare_function):
        self.data_points = data_points
        self.prepare_function = prepare_function

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        feasible = []
        num = 0
        for nodes in data_point.feasible:
            if len(nodes)==3:
                feasible.append(nodes)
                num += 1
                if num >= 100:
                    break
        infeasible = []
        num = 0
        for nodes in data_point.infeasible:
            if len(nodes)==3:
                infeasible.append(nodes)
                num += 1
                if num >= 100:
                    break      
        data_point.feasible = feasible
        data_point.infeasible = infeasible
        nen_sp, e2n_sp, subg_sp, subg_feasibility, node_feats, edge_feats = self.prepare_function(data_point)
        
        return (
            nen_sp,
            e2n_sp,
            subg_sp,
            subg_feasibility,
            node_feats,
            edge_feats
        )
    
args = parser.parse_args()
epochs = args.EPOCH
# Load the data points
print("Load data...")
data_points = load_data_points(args.DATA)

# Use the DataLoader for batch processing
dataset = GraphDataset(data_points, prepare_mean_field)
# Split dataset into training and testing
dataset_size = len(dataset)
train_size = int(args.TRAIN_SPLIT * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
def custom_collate_fn(batch, device):
    """Collate function that keeps sparse tensors separate for each sample."""
    n2n_batch, e2n_batch, subg_batch = [], [], []
    subg_feasibility_batch, node_feats_batch, edge_feats_batch = [], [], []

    for nen_sp, e2n_sp, subg_sp, subg_feasibility, node_feats, edge_feats in batch:
        n2n_batch.append(nen_sp.coalesce().to(device))  # Keep individual sparse tensors
        e2n_batch.append(e2n_sp.coalesce().to(device))
        subg_batch.append(subg_sp.coalesce().to(device))

        subg_feasibility_batch.append(subg_feasibility.to(device))
        node_feats_batch.append(node_feats.to(device))
        edge_feats_batch.append(edge_feats.to(device))

    return n2n_batch, e2n_batch, subg_batch, subg_feasibility_batch, node_feats_batch, edge_feats_batch

train_loader = DataLoader(train_dataset, batch_size=args.BATCH, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, device))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, device))

print("Data loaded. Set up training model ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Binary Cross-Entropy with Logits (applies sigmoid internally)
loss_fn = nn.BCEWithLogitsLoss()

# Initialize the model, optimizer
model = EmbedMeanField(
    latent_dim=args.LATENT_DIMENSION,
    output_dim=1,
    num_node_feats=args.NODE_FEATURE,
    num_edge_feats=args.EDGE_FEATURE,
    max_lv=args.NUM_EMBEDDING)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.LR)

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
        "architecture": "S2V_mean_field",
        "dataset": "data_large",
        "epochs": args.EPOCH,
        }
    )

model.train()  # Set the model to training mode
precision = Precision(task="binary")  # Micro for binary
recall = Recall(task="binary")

for epoch in range(epochs):
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for batch in train_loader:
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_correct = 0
        batch_total = 0
        
        # Process each graph individually
        for nen_sp, e2n_sp, subg_sp, subg_feasibility, node_feats, edge_feats in zip(*batch):
            # Forward pass
            outputs = model(nen_sp, e2n_sp, subg_sp, node_feats, edge_feats)  # (n,)

            # Split indices based on the label
            indices_label_1 = (subg_feasibility == 1).nonzero(as_tuple=True)[0]
            indices_label_0 = (subg_feasibility == 0).nonzero(as_tuple=True)[0]

            # Randomly sample indices from each group
            sampled_indices_1 = indices_label_1[torch.randperm(len(indices_label_1))[:100]]
            sampled_indices_0 = indices_label_0[torch.randperm(len(indices_label_0))[:100]]

            # Select sampled outputs and corresponding target values
            indices = torch.cat([sampled_indices_1, sampled_indices_0])
            sampled_outputs = outputs#[indices]
            sampled_targets = subg_feasibility#[indices]  # Ensure labels are sampled similarly

            # print("Logits:", F.sigmoid(sampled_outputs).squeeze()[:10])

            # Compute loss
            loss = loss_fn(sampled_outputs.squeeze(), sampled_targets)
            batch_loss += loss

            # Compute accuracy
            predictions = torch.sigmoid(sampled_outputs) > args.THRESHOLD  # Threshold
            batch_correct += (predictions.squeeze() == sampled_targets).sum().item()
            batch_total += len(sampled_targets)   

            # Compute precison recall
            precision(sampled_outputs.squeeze(), sampled_targets)
            recall(sampled_outputs.squeeze(), sampled_targets)

        # Backward pass and optimizer step
        batch_loss.backward()

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
        total_loss += batch_loss.item()
        correct_preds += batch_correct
        total_samples += batch_total

    # Calculate epoch accuracy
    train_accuracy = correct_preds / total_samples

    # Evaluate on test set after each epoch
    test_loss, test_accuracy, feasible_sample = evaluate_model(model, test_loader, loss_fn, args.THRESHOLD)

    if args.WANDB:
        wandb.log({"Train Loss": round(total_loss/len(train_loader),4), "Train Accuracy": round(train_accuracy,4),
            "Test Loss": round(test_loss/len(test_loader),4), "Test Accuracy": round(test_accuracy/len(test_loader),4),
            "Precision": precision.compute(), "Recall": recall.compute()})

    print(f"Epoch [{epoch+1}/{epochs}], ", f"Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}",
        f"Precision: {precision.compute():.4f}", f"Recall: {recall.compute():.4f}",
          f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy/len(test_loader):.4f}, Test feasible proportion: {feasible_sample:.4f}")
    precision.reset()
    recall.reset()