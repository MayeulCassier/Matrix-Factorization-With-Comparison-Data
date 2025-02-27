# all imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
import os
import subprocess
import webbrowser
import itertools
import matplotlib.pyplot as plt

############################################
# This file contains Helper functions for matrix factorization
############################################
"""
This is a parameter scan function that will run the experiments over multiple hyperparameter configurations
If a parameter is given as a list, it will iterate over all combinations.
If a parameter is given as a single value, it will be fixed across all experiments.
The ouputs are stored in a dictionary in the format:
[
    {
        'params': {
            'n': 100,
            'm': 200,
            'd': 20,
            'p': 0.5,
            'lr': 0.001,
            'weight_decay': 0.01,
            'num_epochs': 50,
            'reps': 3
        },
        'results': {
            'reconstruction_errors': [0.12, 0.15, 0.14],  # Une valeur par répétition
            'log_likelihoods': [-1.3, -1.4, -1.35],
            'accuracy': [0.82, 0.84, 0.83],
            'gt_log_likelihoods': [-1.2, -1.1, -1.15],
            'gt_accuracy': [0.85, 0.86, 0.87],
            'train_losses': [[0.5, 0.4, 0.3], [0.52, 0.42, 0.32], [0.51, 0.41, 0.31]],  # Une liste des pertes par epoch et par rep
            'val_losses': [[0.55, 0.45, 0.35], [0.56, 0.46, 0.36], [0.54, 0.44, 0.34]]  # Pareil pour validation
        }
    },
    {
        'params': {
            'n': 100,
            'm': 200,
            ...
]
"""
def parameter_scan(n=100, m=200, d=3, p=0.5, s=2.0, device='cpu', 
                    lr=5e-4, weight_decay=5e-3, num_epochs=100, reps=5):
    """
    Runs experiments over multiple hyperparameter configurations.
    If a parameter is given as a list, it will iterate over all combinations.
    If a parameter is given as a single value, it will be fixed across all experiments.

    Parameters:
    - n (int or list): Number of users.
    - m (int or list): Number of items.
    - d (int or list): Latent dimension size.
    - p (float or list): Proportion of user-item interactions used as datapoints.
    - s (float): Scaling factor for preference scores.
    - device (str): Computation device ('cpu' or 'cuda').
    - lr (float or list): Learning rate.
    - weight_decay (float or list): Weight decay for regularization.
    - num_epochs (int or list): Number of training epochs.
    - reps (int or list): Number of repetitions per experiment.

    Returns:
    - dict: Results of each experiment configuration.
    """
    
    # Convert scalar values to lists for iteration
    param_dict = {'n': n, 'm': m, 'd': d, 'p': p, 'lr': lr, 
                  'weight_decay': weight_decay, 'num_epochs': num_epochs, 'reps': reps}
    
    for key, value in param_dict.items():
        if not isinstance(value, (list, tuple)):
            param_dict[key] = [value]  # Wrap single values in a list
    
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*param_dict.values()))
    
    # Store results
    all_results = []
    
    for params in param_combinations:
        param_set = dict(zip(param_dict.keys(), params))
        print(f"\nRunning experiment with parameters: {param_set}")
        
        results = run_experiment(
            n=param_set['n'], m=param_set['m'], d=param_set['d'], p=param_set['p'], 
            s=s, device=device, lr=param_set['lr'], 
            weight_decay=param_set['weight_decay'], reps=param_set['reps'], num_epochs=param_set['num_epochs']
        )
        
        all_results.append({'params': param_set, 'results': results})
    
    return all_results

# Principal function to run the experiments
def run_experiment(n, m, d, p, s, device, lr, weight_decay, reps=5, num_epochs=100):
    """
    Runs multiple experiments for matrix factorization with BTL preference data.

    Parameters:
    - n (int): Number of users.
    - m (int): Number of items.
    - d (int): Latent dimension size.
    - p (float): Proportion of user-item interactions used as datapoints.
    - s (float): Scaling factor for preference scores.
    - device (torch.device): The device where computations will occur.
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay for regularization.
    - reps (int): Number of repetitions for the experiment.
    - num_epochs (int): Number of epochs for training.

    Returns:
    - dict: Contains reconstruction errors, log likelihoods, accuracy,
           ground truth log likelihoods, and ground truth accuracy.
    """
    reconstruction_errors, log_likelihoods, accuracy = [], [], []
    gt_accuracy, gt_log_likelihoods, train_losses, val_losses = [], [], [], []

    for rep in range(reps):
        print(f"\n### Experiment {rep+1}/{reps} started... ###")

        # Step 1: Generate embeddings
        U, V = generate_embeddings(n, m, d, device)

        # Step 2: Create preference dataset
        dataset = BTLPreferenceDataset(U, V, int(n * m * p / 2), scale=s)

        # Step 3: Split dataset into train, validation, and test sets
        train_loader, val_loader, test_loader = split_dataset(dataset, len(dataset))

        # Step 4: Initialize the matrix factorization model
        model = MatrixFactorization(n, m, d).to(device)

        # Step 5: Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Step 6: Train the model and retrieve losses
        t_losses, v_losses = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, is_last=(rep == reps-1))
        train_losses.append(t_losses)
        val_losses.append(v_losses)

        # Step 7: Evaluate the model on the test set
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)

        # Step 8: Compute reconstruction error
        reconstruction_error = compute_reconstruction_error(model, U, V, s)

        # Step 9: Compute ground truth metrics
        gt_loss, gt_acc = compute_ground_truth_metrics(test_loader, U, V, device)

        reconstruction_errors.append(reconstruction_error)
        log_likelihoods.append(-test_loss)
        accuracy.append(test_accuracy)
        gt_log_likelihoods.append(-gt_loss)
        gt_accuracy.append(gt_acc)

        print(f"### Experiment {rep+1}/{reps} completed ###")

    return {
        "reconstruction_errors": reconstruction_errors,
        "log_likelihoods": log_likelihoods,
        "accuracy": accuracy,
        "gt_log_likelihoods": gt_log_likelihoods,
        "gt_accuracy": gt_accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses
    }

############################################
# Structure here is: 
############################################
class BTLPreferenceDataset(Dataset):
    """
    Bradley-Terry-Luce (BTL) preference dataset for training matrix factorization models.

    This dataset generates user-item preference comparisons based on the BTL model.

    Parameters:
    U (torch.Tensor): User embeddings matrix (size: n_users x d).
    V (torch.Tensor): Item embeddings matrix (size: m_items x d).
    num_datapoints (int): Number of preference samples to generate.
    scale (float, optional): Scaling factor for preference scores (default: 1.0).

    Methods:
    - generate_preferences(num_datapoints): Generates pairwise preference data.
    - __len__(): Returns the number of datapoints.
    - __getitem__(idx): Retrieves a specific datapoint.

    Returns:
    - A PyTorch Dataset containing tuples (user_id, item_i, item_j, preference_label).
    """

    def __init__(self, U, V, num_datapoints, scale=1.0):
        self.U = U  # User embeddings
        self.V = V  # Item embeddings
        self.scale = scale  # Scaling factor for preference scores
        self.data = self.generate_preferences(num_datapoints)  # Generate dataset

    def generate_preferences(self, num_datapoints):
        """
        Generates pairwise preferences based on the Bradley-Terry-Luce (BTL) model.

        Parameters:
        num_datapoints (int): Number of preference samples to generate.

        Returns:
        list: A list of tuples (user_id, item_i, item_j, preference_label).
        """

        n_users, d = self.U.shape  # Number of users and embedding dimension
        m_items, _ = self.V.shape  # Number of items
        data = []

        for _ in range(num_datapoints):
            # Randomly select a user
            u = torch.randint(0, n_users, (1,)).item()

            # Randomly select two different items
            i, j = torch.randint(0, m_items, (2,)).tolist()
            while i == j:  # Ensure i and j are not the same item
                j = torch.randint(0, m_items, (1,)).item()

            # Compute preference score using the BTL model
            user_embedding = self.U[u]  # Retrieve user embedding
            item_diff = self.V[i] - self.V[j]  # Compute item difference
            preference_score = torch.sigmoid(self.scale * torch.dot(user_embedding, item_diff))

            # Assign label: 1 if user prefers item i over j, else 0
            label = torch.bernoulli(preference_score).item()  # Sample from Bernoulli distribution
            data.append((u, i, j, label))

        return data

    def __len__(self):
        """Returns the total number of datapoints in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a specific preference datapoint."""
        return self.data[idx]

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model using user and item embeddings with dropout regularization.

    This model is based on the Bradley-Terry-Luce (BTL) preference model, where a user 
    is given two items (i and j), and the model predicts the probability that the user 
    prefers item i over item j.

    Parameters:
    n_users (int): Number of users.
    n_items (int): Number of items.
    d (int): Latent dimension size for embeddings.

    Methods:
    - forward(u, i, j): Computes the preference probability between two items for a user.

    Returns:
    - A probability score (between 0 and 1) indicating the preference of item i over item j.
    """

    def __init__(self, n_users, n_items, d):
        super(MatrixFactorization, self).__init__()

        # Initialize user and item embeddings with normalized random values
        self.U = nn.Parameter(torch.randn(n_users, d) / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
        self.V = nn.Parameter(torch.randn(n_items, d) / torch.sqrt(torch.tensor(d, dtype=torch.float32)))

    def forward(self, u, i, j):
        """
        Forward pass for computing preference scores using the BTL model.

        Parameters:
        u (torch.Tensor): Tensor of user indices.
        i (torch.Tensor): Tensor of first item indices.
        j (torch.Tensor): Tensor of second item indices.

        Returns:
        torch.Tensor: Probability score indicating preference of item i over item j.
        """

        # Retrieve user and item embeddings with dropout applied
        u_embedding = self.U[u]  # User embedding
        i_embedding = self.V[i]  # Item i embedding
        j_embedding = self.V[j]  # Item j embedding

        # Compute the difference between items i and j
        diff = torch.sum(u_embedding * (i_embedding - j_embedding), dim=1)

        # Apply sigmoid activation to get a probability score (between 0 and 1)
        return torch.sigmoid(diff)





############################################
# Now the Useful functions:
############################################


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, is_last=False):
    """
    Trains a matrix factorization model using binary cross-entropy loss.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    train_loader (DataLoader): DataLoader for the training dataset.
    val_loader (DataLoader): DataLoader for the validation dataset.
    optimizer (torch.optim.Optimizer): Optimizer used for training.
    device (torch.device): Device (CPU or GPU) where training will occur.
    num_epochs (int, optional): Number of epochs for training (default: 100).

    Returns:
    tuple: Lists containing training loss and validation loss per epoch.
    """
    
    # Define logging directory for TensorBoard and reset previous logs
    if is_last:
        log_dir = 'runs/matrix_factorization'
        start_tensorboard(log_dir=log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        total_loss = 0
        model.train()  # Set model to training mode
    
        # Iterate through training batches
        for batch in train_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Move batch data to device
            optimizer.zero_grad()  # Reset gradients
            pred = model(u, i, j)  # Forward pass
            loss = F.binary_cross_entropy(pred, z.float())  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_loss += loss.item()  # Accumulate loss
    
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
    
        # Log training loss to TensorBoard
        if is_last:
            writer.add_scalar('Loss/train', train_loss, epoch)
    
        # Validation phase
        val_loss = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_loader:
                u, i, j, z = [x.to(device) for x in batch]
                pred = model(u, i, j)
                loss = F.binary_cross_entropy(pred, z.float())
                val_loss += loss.item()
    
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
    
        # Log validation loss to TensorBoard
        if is_last:
            writer.add_scalar('Loss/val', val_loss, epoch)
    
    if is_last:
        writer.close()  # Close TensorBoard writer
    
    return train_losses, val_losses

        


def evaluate_model(model, test_loader, device):
    """
    Evaluates a trained model on the test dataset.

    Parameters:
    model (torch.nn.Module): The trained model to evaluate.
    test_loader (DataLoader): DataLoader for the test dataset.
    device (torch.device): Device (CPU or GPU) where evaluation will occur.

    Returns:
    tuple: Average test loss and accuracy of the model.
    """

    model.eval()  # Set model to evaluation mode
    test_loss, correct, total = 0, 0, 0  # Initialize tracking variables

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in test_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Move batch data to device

            # Forward pass
            pred = model(u, i, j)
            loss = F.binary_cross_entropy(pred, z.float())  # Compute loss
            test_loss += loss.item()

            # Convert probabilities to binary predictions (threshold at 0.5)
            pred = (pred > 0.5).float()

            # Count correct predictions
            correct += (pred == z).sum().item()
            total += len(z)

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    return test_loss / len(test_loader), accuracy


def compute_reconstruction_error(model, U, V, scale):
    """
    Computes the reconstruction error of the matrix factorization model.

    Parameters:
    model (torch.nn.Module): The trained model containing learned user (U) and item (V) embeddings.
    U (torch.Tensor): Ground truth user embeddings (size: n x d).
    V (torch.Tensor): Ground truth item embeddings (size: m x d).
    scale (float): Scaling factor applied to the ground truth embeddings.

    Returns:
    float: Mean squared error (MSE) between the reconstructed matrix and the ground truth.
    """

    # Compute the reconstructed user-item matrix from learned embeddings
    user_item_matrix = torch.mm(model.U, model.V.t())

    # Center the reconstructed matrix by subtracting the mean of each row
    user_item_matrix -= torch.mean(user_item_matrix, dim=1, keepdim=True)

    # Compute Mean Squared Error (MSE) between reconstructed and scaled ground truth matrix
    reconstruction_error = F.mse_loss(user_item_matrix, scale * torch.mm(U, V.t()))

    return reconstruction_error.item()  # Return as a Python float

def compute_ground_truth_metrics(test_loader, U, V, device):
    """
    Computes ground truth metrics (MSE loss and accuracy) using pre-trained embeddings.

    Parameters:
    test_loader (DataLoader): DataLoader for the test dataset.
    U (torch.Tensor): Ground truth user embeddings (size: n x d).
    V (torch.Tensor): Ground truth item embeddings (size: m x d).
    device (torch.device): Device (CPU or GPU) where computation will occur.

    Returns:
    tuple: Mean squared error (MSE) loss and accuracy based on ground truth embeddings.
    """

    gt_loss, gt_correct, gt_total = 0, 0, 0  # Initialize tracking variables

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in test_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Move batch data to device

            # Retrieve user embeddings and compute item difference
            user_embeddings = U[u]  # Select user embeddings based on indices
            item_diff = V[i] - V[j]  # Compute item pairwise differences

            # Compute probability using sigmoid function
            gt_prob = torch.sigmoid(torch.sum(user_embeddings * item_diff, dim=1))

            # Compute Mean Squared Error (MSE) loss
            loss = F.mse_loss(gt_prob, z.float())
            gt_loss += loss.item()

            # Convert predictions to binary labels (threshold at 0)
            gt_pred = (torch.sum(user_embeddings * item_diff, dim=1) > 0).float()

            # Count correct predictions
            gt_correct += (gt_pred == z).sum().item()
            gt_total += len(z)

    # Compute accuracy
    gt_accuracy = gt_correct / gt_total if gt_total > 0 else 0.0

    return gt_loss / len(test_loader), gt_accuracy



def split_dataset(dataset, num_datapoints, train_ratio=0.8, val_ratio=0.1, batch_size=64):
    """
    Splits a dataset into training, validation, and test sets and creates DataLoaders.
    - Split by random_split

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to be split.
    num_datapoints (int): Total number of datapoints in the dataset.
    train_ratio (float, optional): Proportion of data to use for training (default: 0.8).
    val_ratio (float, optional): Proportion of data to use for validation (default: 0.1).
    batch_size (int, optional): Batch size for DataLoader (default: 64).

    Returns:
    tuple: DataLoaders for training, validation, and test sets.
    """

    # Compute dataset sizes for each split
    train_size = int(train_ratio * num_datapoints)
    val_size = int(val_ratio * num_datapoints)
    test_size = num_datapoints - train_size - val_size  # Ensure all datapoints are used

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each dataset split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def start_tensorboard(log_dir='runs/matrix_factorization', port=6006):
    """Starts TensorBoard and opens it in the default web browser."""
    # Remove old logs
    shutil.rmtree(log_dir, ignore_errors=True)
    
    # Launch TensorBoard in the background
    subprocess.Popen(["tensorboard", "--logdir=runs", f"--port={port}"], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for TensorBoard to start
    time.sleep(3)
    
    # Open TensorBoard in the browser
    # webbrowser.open(f"http://localhost:{port}/")
    # print(f"🔥 TensorBoard launched at http://localhost:{port}/")


############################################
# These Funtions create U and V with different structures
############################################

def generate_embeddings(n, m, d, device):
    """
    Generates random user (U) and item (V) embeddings for matrix factorization.
    - Normal distribution (mean 0, variance 1) for U and V

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    device (torch.device): The device where tensors will be stored (CPU/GPU).

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """
    
    # Convert d to a tensor for numerical stability
    d_tensor = torch.tensor(d, dtype=torch.float32)

    # Initialize user embeddings (U) and item embeddings (V) with random values
    # Normalize by sqrt(d) to keep variance stable
    U = torch.randn(n, d, device=device) / torch.sqrt(d_tensor) # n x d
    V = torch.randn(m, d, device=device) / torch.sqrt(d_tensor) # m x d

    # Center the item embeddings by subtracting the mean per row
    V -= torch.mean(V, dim=1, keepdim=True)

    return U, V

def generate_structured_embeddings(n, m, d, num_clusters=5, cluster_std=0.1, device="cpu"):
    """
    Generates structured user (U) and item (V) embeddings with predefined clusters.
    
    Instead of random initialization, this method ensures that items are grouped into clusters, 
    and users have preferences that align with these clusters.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    num_clusters (int): Number of clusters for items.
    cluster_std (float): Standard deviation for item dispersion within clusters.
    device (str): "cpu" or "cuda" to specify where the tensors should be stored.

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """
    
    # Create cluster centers for items
    cluster_centers = torch.randn(num_clusters, d, device=device)

    # Assign each item to a cluster with a small random perturbation
    item_cluster_assignments = torch.randint(0, num_clusters, (m,))
    V = cluster_centers[item_cluster_assignments] + cluster_std * torch.randn(m, d, device=device)

    # Generate user embeddings as a combination of cluster preferences
    user_cluster_affinity = torch.randn(n, num_clusters, device=device)  # User affinity to clusters
    U = torch.matmul(user_cluster_affinity, cluster_centers)  # Compute user embeddings based on cluster proximity

    return U, V


def generate_svd_embeddings(n, m, d, noise_level=0.1, device="cpu"):
    """
    Generates structured embeddings using an underlying latent score matrix and SVD decomposition.
    
    Instead of initializing U and V randomly, this method creates an underlying user-item 
    preference matrix and decomposes it into latent factors.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    noise_level (float): Amount of noise added to the true score matrix to prevent overfitting.
    device (str): "cpu" or "cuda" to specify where the tensors should be stored.

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """
    
    # Generate a structured latent score matrix
    true_score_matrix = torch.randn(n, m, device=device)

    # Perform Singular Value Decomposition (SVD) to extract latent factors
    U, S, Vt = torch.svd(true_score_matrix)

    # Keep only the top d components to obtain dense embeddings
    U = U[:, :d] * torch.sqrt(S[:d])  # Scale by singular values
    V = Vt[:d, :].T * torch.sqrt(S[:d])  # Transpose to match expected dimensions

    # Add a small amount of noise to avoid perfect factorization
    U += noise_level * torch.randn_like(U)
    V += noise_level * torch.randn_like(V)

    return U.to(device), V.to(device)


def generate_correlated_embeddings(n, m, d, correlation_factor=0.8, device="cpu"):
    """
    Generates user and item embeddings where dimensions are correlated.

    By default, random embeddings have independent dimensions. This method introduces 
    correlation between dimensions, making learning more structured.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    correlation_factor (float): Degree of correlation between dimensions (0: none, 1: fully correlated).
    device (str): "cpu" or "cuda" to specify where the tensors should be stored.

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """
    
    # Generate base embeddings for users and items
    U = torch.randn(n, d, device=device)
    V = torch.randn(m, d, device=device)

    # Create a correlation matrix
    correlation_matrix = torch.eye(d, device=device) * (1 - correlation_factor) + correlation_factor * torch.ones((d, d), device=device)
    
    # Apply the correlation transformation
    U = U @ correlation_matrix
    V = V @ correlation_matrix

    return U, V


import networkx as nx

def generate_graph_embeddings(n, m, d, device="cpu"):
    """
    Generates user (U) and item (V) embeddings using a graph-based structure.

    Users and items are connected in a bipartite graph. The embeddings are derived from 
    a graph-based transformation, ensuring items and users with similar interactions 
    have nearby embeddings.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    device (str): "cpu" or "cuda".

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """

    # Create a bipartite graph (users and items)
    G = nx.Graph()
    
    # Add nodes for users and items
    G.add_nodes_from(range(n), bipartite=0)  # User nodes
    G.add_nodes_from(range(n, n + m), bipartite=1)  # Item nodes

    # Randomly connect users to items (simulating interactions)
    for u in range(n):
        num_connections = torch.randint(10, 20, (1,)).item()  # Each user interacts with 1-5 items
        item_choices = torch.randint(n, n + m, (num_connections,)).tolist()
        G.add_edges_from([(u, item) for item in item_choices])

    # Compute node embeddings using Spectral Embedding
    spectral_embeddings = nx.spectral_layout(G, dim=d)

    # Extract user and item embeddings
    U = torch.tensor([spectral_embeddings[i] for i in range(n)], dtype=torch.float32, device=device)
    V = torch.tensor([spectral_embeddings[i] for i in range(n, n + m)], dtype=torch.float32, device=device)

    return U, V

import networkx as nx

def generate_social_embeddings(n, m, d, social_influence=0.3, device="cpu"):
    """
    Generates user (U) and item (V) embeddings influenced by social connections.

    Users who are connected have similar embeddings, reflecting real-world social influences.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    social_influence (float): Strength of social similarity.
    device (str): "cpu" or "cuda".

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """

    # Generate base embeddings
    U = torch.randn(n, d, device=device)
    V = torch.randn(m, d, device=device)

    # Create a social network graph (small-world model)
    G = nx.watts_strogatz_graph(n, k=5, p=0.1)

    # Adjust user embeddings based on social similarity
    for u in range(n):
        friends = list(G.neighbors(u))
        if friends:
            friend_embeddings = torch.stack([U[f] for f in friends]).mean(dim=0)
            U[u] = (1 - social_influence) * U[u] + social_influence * friend_embeddings

    return U, V


def generate_temporal_embeddings(n, m, d, timesteps=5, device="cpu"):
    """
    Generates user (U) and item (V) embeddings that evolve over time.

    Users and items have a base latent factor, but their preferences shift gradually, 
    simulating real-world dynamics.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    timesteps (int): Number of time steps to simulate.
    device (str): "cpu" or "cuda".

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """

    # Base embeddings
    U_base = torch.randn(n, d, device=device)
    V_base = torch.randn(m, d, device=device)

    # Temporal drift (each time step shifts embeddings slightly)
    time_drift = torch.randn(n, d, device=device) * 0.02  # Small changes per step
    U = U_base + timesteps * time_drift

    time_drift_items = torch.randn(m, d, device=device) * 0.02
    V = V_base + timesteps * time_drift_items

    return U, V

def generate_hierarchical_embeddings(n, m, d, num_groups=5, device="cpu"):
    """
    Generates structured embeddings where users belong to hierarchical groups.

    Users inherit a portion of their embeddings from their assigned group, 
    adding structured dependencies.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    num_groups (int): Number of user groups.
    device (str): "cpu" or "cuda".

    Returns:
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """

    # Generate base group embeddings
    group_embeddings = torch.randn(num_groups, d, device=device)

    # Assign users to groups
    user_group_assignments = torch.randint(0, num_groups, (n,))

    # Generate users based on group embeddings + slight individual variation
    U = group_embeddings[user_group_assignments] + 0.1 * torch.randn(n, d, device=device)

    # Generate structured item embeddings
    V = torch.randn(m, d, device=device)  # Items remain independent

    return U, V

from sklearn.mixture import GaussianMixture

def generate_gmm_embeddings(n, m, d, num_clusters=5, device="cpu"):
    """
    Generates structured user (U) and item (V) embeddings using Gaussian Mixture Models (GMM).

    Users and items belong to multiple clusters but with varying probabilities,
    creating a more natural embedding space.

    Parameters:
    n (int): Number of users.
    m (int): Number of items.
    d (int): Latent dimension (embedding size).
    num_clusters (int): Number of clusters for the mixture model.
    device (str): "cpu" or "cuda".

    Returns:y
    tuple: User embeddings U (n x d) and item embeddings V (m x d).
    """
    
    # Fit Gaussian Mixture Model to generate cluster centers
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    
    # Generate cluster centers for users and items
    user_clusters = gmm.fit_predict(torch.randn(n, d).numpy())
    item_clusters = gmm.fit_predict(torch.randn(m, d).numpy())

    # Create structured embeddings
    U = torch.tensor(gmm.means_[user_clusters], dtype=torch.float32, device=device)
    V = torch.tensor(gmm.means_[item_clusters], dtype=torch.float32, device=device)

    return U, V


############################################
# These functions are used to visualize the results
############################################

# Plotting the losses
def plot_losses(results, param_index=None):
    """
    Plots training and validation losses.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_index (int, optional): The index of a specific experiment to plot train & val losses on the same graph.
      If None, plots all training losses in one graph and all validation losses in another.
    """
    if param_index is not None:
        # Plot a specific experiment's train and validation loss on the same graph
        exp = results[param_index]
        train_losses = exp['results']['train_losses']
        val_losses = exp['results']['val_losses']
        
        plt.figure(figsize=(10, 5))
        for rep in range(len(train_losses)):
            plt.plot(train_losses[rep], label=f'Train Loss (rep {rep+1})', linestyle='--')
            plt.plot(val_losses[rep], label=f'Val Loss (rep {rep+1})')
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train & Val Loss for {exp['params']}")
        plt.legend()
        plt.show()
    else:
        # Plot all train losses in one graph
        plt.figure(figsize=(10, 5))
        for i, exp in enumerate(results):
            for rep in range(len(exp['results']['train_losses'])):
                plt.plot(exp['results']['train_losses'][rep], label=f'Exp {i+1}, Rep {rep+1}')
        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title("Training Losses for All Experiments")
        plt.legend()
        plt.show()

        # Plot all validation losses in one graph
        plt.figure(figsize=(10, 5))
        for i, exp in enumerate(results):
            for rep in range(len(exp['results']['val_losses'])):
                plt.plot(exp['results']['val_losses'][rep], label=f'Exp {i+1}, Rep {rep+1}')
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title("Validation Losses for All Experiments")
        plt.legend()
        plt.show()