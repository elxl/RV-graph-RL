import argparse
import warnings
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean, scatter_std
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Precision, Recall
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.utils.pytorch_util import add_feature_initialization
from src.net.s2v_subtour import Struc2Vec
from src.utils.pytorch_util import evaluate_model_s2v
from multiprocessing import Pool, cpu_count
import pickle

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning
)

parser = argparse.ArgumentParser(description="s2v training.")
parser.add_argument(
    '--data',
    type=str,
    default='./data/training/data_0-1_full_10000_di_edge.pt',
    help="Directory for training data."
)
parser.add_argument(
    '--p_dim',
    type=int,
    default=16,
    help="Node feature dimension. (default: 16)"
)
parser.add_argument(
    '--LR',
    type=float,
    default=0.001,
    help="Learning rate. (default: 0.001)"
)
parser.add_argument(
    '--iteration',
    type=int,
    default=4,
    help="Number of iterations for embedding. (default: 4)"
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help="Batch size for training. (default: 32)"
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help="Threshold for binary classification. (default: 0.5)"
)
parser.add_argument(
    '--epoch',
    type=int,
    default=10,
    help="Number of training epochs. (default: 20)"
)
parser.add_argument(
    '--save_weights',
    type=int,
    default=0,
    help="Save weights for the model. (default: 0)"
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    default='./weights/s2v/saved_weights.pt',
    help="Path to the checkpoint file."
)
parser.add_argument(
    '--verbose',
    type=int,
    default=1,
    help="Print training progress. (default: 1)"
)
parser.add_argument(
    '--wandb',
    type=int,
    default=0,
    help="Either using wandb or not."
)


# Add intitial features for each data point
if __name__ == '__main__':
    args = parser.parse_args()
    p_dim = args.p_dim
    r = args.iteration
    batch_size = args.batch_size
    EPOCH = args.epoch
    THRESHOLD = args.threshold
    LR = args.LR
    VERBOSE = args.verbose
    SAVE_WEIGHTS = args.save_weights

    data_lists = torch.load(args.data)

    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="tour-feasibility",

            # track hyperparameters and run metadata
            config={
            "p_dim": p_dim,
            "learning_rate": LR,
            "iterations": r,
            "batch_size": batch_size,
            "architecture": "s2v",
            "dataset": args.data,
            }
        )

    with Pool(cpu_count()) as p:
        data = p.starmap(add_feature_initialization, [(data, p_dim) for data in data_lists])

    # Split dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

    # Create DataLoader for batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Set up training model ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Binary Cross-Entropy with Logits
    loss_fn = nn.BCELoss(reduction='sum')

    # Initialize the model, optimizer
    model = Struc2Vec(
        p_dim=p_dim,
        nfeatures_vehicle=2,
        nfeatures_pickup=2,
        nfeatures_dropoff=2,
        nfeatures_edge=2,
        r=r
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training
    print("Training start...")


    precision = Precision(task="binary")  # Micro for binary
    recall = Recall(task="binary")

    for epoch in range(EPOCH):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        total_data_points = 0
        best_test_loss = np.inf
        all_preds = []
        all_labels = []
        
        # mislabled data
        mislabeled = False
        mislabeled_labels_train = []
        mislabeled_scores_train = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

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

            # print("Logits:", F.sigmoid(sampled_outputs).squeeze()[:10])

            # Compute loss
            loss = loss_fn(outputs, batch.y.view(-1, 1).float())

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
            total_loss += loss.item()* batch.y.size(0)
            total_data_points += batch.y.size(0)

            preds = (outputs > THRESHOLD).float()  # Convert logits to binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            precision.update(preds, batch.y.view(-1, 1))
            recall.update(preds, batch.y.view(-1, 1))

            # Save mislabed data
            if epoch == EPOCH - 1:
                for i,pred in enumerate(preds):
                    if pred.item() != batch.y[i].item():
                        mislabeled_labels_train.append(batch.y[i].cpu().item())
                        mislabeled_scores_train.append(outputs[i].cpu().item())

        # Calculate epoch accuracy and average loss
        train_accuracy = accuracy_score(all_labels, all_preds)
        average_train_loss = total_loss / total_data_points

        # Evaluate on test set after each epoch
        with torch.no_grad():
            if epoch == EPOCH - 1:
                mislabeled = True
            average_test_loss, test_accuracy, mislabeled_labels_test, mislabeled_scores_test = evaluate_model_s2v(model, test_loader, loss_fn, THRESHOLD, device, mislabeled)

        torch.cuda.empty_cache()

        if args.wandb:
            # sizes = correct_size.keys()
            # for size in sizes:
            #     all_sample = correct_size[size] + incorrect_size[size]
            #     wandb.log({f"Correct {size}": correct_size[size]/all_sample, f"Incorrect {size}": incorrect_size[size]/all_sample})
            # with open("training_log.txt", "a") as log_file:
            #     log_file.write(f"Epoch [{epoch+1}/{EPOCH}]\n")
            #     sizes = correct_size.keys()
            #     for size in sizes:
            #         all_sample = correct_size[size] + incorrect_size[size]
            #         correct_ratio = correct_size[size] / all_sample if all_sample > 0 else 0
            #         log_file.write(f"Size {size}: Correct Ratio: {correct_ratio:.2f}\n")
            wandb.log({"Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy, "Train Loss": average_train_loss, "Test Loss": average_test_loss,
                      "Precision": precision.compute(), "Recall": recall.compute()})

        if VERBOSE:
            print(f"Epoch [{epoch+1}/{EPOCH}], ", f"Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}",
                f"Precision: {precision.compute():.4f}", f"Recall: {recall.compute():.4f}",
                f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        precision.reset()
        recall.reset()

        if SAVE_WEIGHTS and average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': average_train_loss,
            'test_loss': average_test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, args.checkpoint_path)