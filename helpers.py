# all imports
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader, random_split, Dataset # type: ignore
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # type: ignore
import shutil
import time
import os
import subprocess
import webbrowser
import itertools
from generation_data import *

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
                    lr=5e-4, weight_decay=5e-3, num_epochs=100, reps=5, open_browser=False):
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
                  'weight_decay': weight_decay, 'num_epochs': num_epochs, 'reps': reps, 's': s}
    
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
            s=param_set['s'], device=device, lr=param_set['lr'], 
            weight_decay=param_set['weight_decay'], reps=param_set['reps'], num_epochs=param_set['num_epochs'], open_browser=open_browser
        )
        
        all_results.append({'params': param_set, 'results': results})
    
    return all_results

# Principal function to run the experiments
def run_experiment(n, m, d, p, s, device, lr, weight_decay, reps=5, num_epochs=100, open_browser=False):
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
        # print(f"\n### Experiment {rep+1}/{reps} started... ###")

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
        t_losses, v_losses = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
                                          is_last=(rep == reps-1), open_browser=open_browser)
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

        # print(f"### Experiment {rep+1}/{reps} completed ###")

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


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, is_last=False, open_browser=False):
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
        start_tensorboard(log_dir=log_dir, open_browser=open_browser)
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
            writer.add_scalars('Losses', {'train': train_loss, 
                              'val': val_loss}, epoch)
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

def start_tensorboard(log_dir='runs/matrix_factorization', port=6006, open_browser=True):
    """Starts TensorBoard and opens it in the default web browser."""
    # Remove old logs
    shutil.rmtree(log_dir, ignore_errors=True)
    
    # Launch TensorBoard in the background
    subprocess.Popen(["tensorboard", "--logdir=runs", f"--port={port}"], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for TensorBoard to start
    time.sleep(3)
    
    # Open TensorBoard in the browser
    if open_browser:
        webbrowser.open(f"http://localhost:{port}/")
        print(f"🔥 TensorBoard launched at http://localhost:{port}/")




