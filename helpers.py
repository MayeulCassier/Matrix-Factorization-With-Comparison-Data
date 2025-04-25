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
                    lr=1e-3, weight_decay=1e-5, num_epochs=100, reps=5, open_browser=False, linear = False, K=1, d1 = None):
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
    - open_browser (bool): Whether to open TensorBoard in the default web browser.
    - linear (bool): Whether to perform a linear scan (synchronized index) or full combination scan.
    - K (int): Number of labels per preference comparison.

    Returns:
    - dict: Results of each experiment configuration.
    """
    
    # Convert scalar values to lists for iteration
    param_dict = {'n': n, 'm': m, 'd': d, 'p': p, 'lr': lr, 
                  'weight_decay': weight_decay, 'num_epochs': num_epochs, 'reps': reps, 's': s, 'K': K, 'd1': d1}
    param_dict = {
        k: list(v) if isinstance(v, np.ndarray) else
           [float(x) if isinstance(x, (np.float32, np.float64)) else int(x) if isinstance(x, np.integer) else x for x in v] 
           if isinstance(v, list) else
           float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, np.integer) else v
        for k, v in param_dict.items()
    }
    # Filter out non-list parameters
    list_params = [v for v in param_dict.values() if isinstance(v, list)]

    # Condition for activating linear scan
    if len(list_params) <= 1:  
        stop = True  # 0 or 1 list → always True
    else:
        stop = all(len(v) == len(list_params[0]) for v in list_params)  # Vérify if all lists have the same length
    
    

    for key, value in param_dict.items():
        if not isinstance(value, (list, tuple)):
            param_dict[key] = [value]  # Wrap single values in a list
    if not linear:
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
                weight_decay=param_set['weight_decay'], reps=param_set['reps'], num_epochs=param_set['num_epochs'], 
                open_browser=open_browser, K=param_set['K'], d1 = param_set['d1']
            )
            
            all_results.append({'params': param_set, 'results': results})
        
        return all_results
    elif linear and stop:
        # Linear scan (synchronized index)
        all_results = []
        for i in range(len(list_params[0])):
            params = {k: v[i] if len(v) > 1 else v[0] for k, v in param_dict.items()}
            print(f"\nRunning experiment with parameters: {params}")
            results = run_experiment(
                n=params['n'], m=params['m'], d=params['d'], p=params['p'], 
                s=params['s'], device=device, lr=params['lr'], 
                weight_decay=params['weight_decay'], reps=params['reps'], num_epochs=params['num_epochs'], 
                open_browser=open_browser, K=params['K'], d1 = params['d1']
            )
            all_results.append({'params': params, 'results': results})
        return all_results
    else:
        raise ValueError("The linear scan is not possible because the parameters are not synchronized.")

# Principal function to run the experiments
def run_experiment(n, m, d, p, s, device, lr, weight_decay, reps=5, num_epochs=100, open_browser=False, K=1, d1=None):
    """
    Runs multiple experiments for matrix factorization with clean BTL preference data.
    Uses disjoint triplet splits to avoid overlap between train and test.

    Parameters:
    - n, m, d, p, s, device, lr, weight_decay, reps, num_epochs, open_browser, K, d1: same as before.

    Returns:
    - dict with metrics from all repetitions.
    """
    reconstruction_errors, log_likelihoods, accuracy = [], [], []
    gt_accuracy, gt_log_likelihoods, train_losses, val_losses = [], [], [], []

    for rep in range(reps):
        if d1 is None:
            d1 = d

        # Step 1: Generate ground truth matrix X (size n x m)
        X = generate_embeddings(n, m, d1, device)

        # Step 2: Generate triplets then split & label cleanly
        num_triplets = int(n * m * p / 2)
        train_loader, val_loader, test_loader = split_dataset_from_triplets(X, num_triplets, scale=s, K=K)

        # Step 3: Initialize model and optimizer
        model = MatrixFactorization(n, m, d).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Step 4: Train the model
        is_last = (rep == reps - 1)
        t_losses, v_losses = train_model(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, is_last=is_last, open_browser=open_browser
        )
        train_losses.append(t_losses)
        val_losses.append(v_losses)

        # Step 5: Evaluate the model
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        accuracy.append(test_acc)
        log_likelihoods.append(-test_loss)

        # Step 6: Compute reconstruction error
        rec_error = compute_reconstruction_error(model, X)
        reconstruction_errors.append(rec_error)

        # Step 7: Ground truth evaluation
        gt_loss, gt_acc = compute_ground_truth_metrics(test_loader, X, device)
        gt_log_likelihoods.append(-gt_loss)
        gt_accuracy.append(gt_acc)

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
    Dataset contenant des préférences BTL, générées à partir d'une liste de triplets (u, i, j)
    avec K répétitions de labels tirés selon la probabilité sigmoïde.
    """
    def __init__(self, triplets, X, scale=1.0, K=1):
        self.X = X
        self.scale = scale
        self.data = self._generate_labels(triplets, K)

    def _generate_labels(self, triplets, K):
        data = []
        for (u, i, j) in triplets:
            score = torch.sigmoid(self.scale * (self.X[u, i] - self.X[u, j]))
            for _ in range(K):
                label = torch.bernoulli(score).item()
                data.append((u, i, j, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def split_dataset_from_triplets(X, num_triplets, scale=1.0, K=1, train_ratio=0.8, val_ratio=0.1, batch_size=64):
    """
    Génère une liste de triplets uniques (u, i, j), fait un split, puis génère des datasets avec K labels.
    Si le test set est trop petit (<500), il est complété automatiquement sans chevauchement avec train/val.
    """

    n, m = X.shape

    # Génère les triplets (u, i, j) uniques
    triplets = set()
    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        i, j = torch.randint(0, m, (2,)).tolist()
        if i != j:
            triplets.add((u, i, j))
    triplets = list(triplets)

    # Shuffle et split
    total = len(triplets)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    train_triplets, val_triplets, test_triplets = torch.utils.data.random_split(
        triplets, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Vérifie et complète le test set si besoin
    MIN_TEST_POINTS = 500
    if len(test_triplets) * K < MIN_TEST_POINTS:
        seen_triplets = set(train_triplets.indices + val_triplets.indices + test_triplets.indices)
        seen_triplets = set([triplets[i] for i in seen_triplets])
        needed = (MIN_TEST_POINTS + K - 1) // K - len(test_triplets)
        extra = set()
        while len(extra) < needed:
            u = torch.randint(0, n, (1,)).item()
            i, j = torch.randint(0, m, (2,)).tolist()
            t = (u, i, j)
            if i != j and t not in seen_triplets and t not in extra:
                extra.add(t)
        test_triplets = list(test_triplets) + list(extra)

    # Génère les datasets
    train_dataset = BTLPreferenceDataset(train_triplets, X, scale=scale, K=K)
    val_dataset = BTLPreferenceDataset(val_triplets, X, scale=scale, K=K)
    test_dataset = BTLPreferenceDataset(test_triplets, X, scale=scale, K=K)

    # Crée les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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
    is_last (bool, optional): Whether this is the last experiment (default: False).
    open_browser (bool, optional): Whether to open TensorBoard in the default web browser (default: False).

    Returns:
    tuple: Lists containing training loss and validation loss per epoch.
    """
    
    # Define logging directory for TensorBoard and reset previous logs
    if False:
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
        if False:
            writer.add_scalars('Losses', {'train': train_loss, 
                              'val': val_loss}, epoch)
    if False:
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

# Reconstruction error
def compute_reconstruction_error(model, X):
    """
    Computes the reconstruction error of the matrix factorization model.

    Parameters:
    model (torch.nn.Module): The trained model containing learned user (U) and item (V) embeddings.
    X (torch.Tensor): Ground truth user-item matrix (size: n x m).
    scale (float): Scaling factor applied to the ground truth embeddings.

    Returns:
    float: Mean squared error (MSE) between the reconstructed matrix and the ground truth.
    """

    # Compute the reconstructed user-item matrix from learned embeddings
    user_item_matrix = torch.mm(model.U, model.V.t())

    # Center the reconstructed matrix by subtracting the mean of each row
    user_item_matrix -= torch.mean(user_item_matrix, dim=0, keepdim=True)

    # Compute the Frobenius norm of the ground truth matrix
    frobenius_norm = torch.norm(X, p="fro")

    # Compute Mean Squared Error (MSE)
    reconstruction_error = torch.norm(user_item_matrix - X, p = "fro")

    # Normalize MSE by the Frobenius norm
    normalized_error = reconstruction_error / frobenius_norm

    return normalized_error.item()  # Return as a Python float

# Ground truth metrics
def compute_ground_truth_metrics(test_loader, X, device):
    """
    Computes ground truth metrics (MSE loss and accuracy) using pre-trained embeddings.

    Parameters:
    test_loader (DataLoader): DataLoader for the test dataset.
    X (torch.Tensor): User-item matrix (size: n x m) containing user-item interactions.
    device (torch.device): Device (CPU or GPU) where computation will occur.

    Returns:
    tuple: Mean squared error (MSE) loss and accuracy based on ground truth embeddings.
    """

    gt_loss, gt_correct, gt_total = 0, 0, 0  # Initialize tracking variables

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in test_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Move batch data to device

            diff = X[u, i] - X[u, j]  # Compute item pairwise differences

            # Compute probability using sigmoid function
            gt_prob = torch.sigmoid(diff)

            # Compute Mean Squared Error (MSE) loss
            loss = F.mse_loss(gt_prob, z.float())
            gt_loss += loss.item()

            # Convert predictions to binary labels (threshold at 0)
            pred = (diff > 0).float()

            # Count correct predictions
            gt_correct += (pred == z).sum().item()
            gt_total += len(z)

    # compute accuracy
    accuracy = gt_correct / gt_total if gt_total > 0 else 0.0

    return gt_loss / len(test_loader), accuracy



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

############################################
# tests for evalute_ground_truth
############################################

# test the function evaluate_ground_truth
def evaluate_ground_truth(n, m, p, d, s, device, K, reps=1):
    """
    Generates embeddings, creates preferences using BTLPreferenceDataset, 
    and computes ground truth accuracy and loss.

    Parameters:
    - n (int): Number of users.
    - m (int): Number of items.
    - p (float): Proportion of user-item interactions used as datapoints.
    - d (int): Latent dimension size.
    - s (float): Scaling factor for preference scores.
    - device (str): Computation device ('cpu' or 'cuda').
    - K (int): Number of labels per preference comparison.
    - reps (int): Number of repetitions for the experiment.

    Returns:
    - tuple: Ground truth loss and accuracy.
    """
    losses = []
    accuracies = []

    for _ in range(reps):
        # Génère les embeddings
        X = generate_embeddings(n, m, d, device)

        # Nombre de triplets à générer
        num_triplets = int(n * m * p / 2)

        # Utilise le split propre basé sur les triplets
        _, _, test_loader = split_dataset_from_triplets(X, num_triplets, scale=s, K=K)

        # Évalue
        gt_loss, gt_acc = compute_ground_truth_metrics(test_loader, X, device)
        losses.append(gt_loss)
        accuracies.append(gt_acc)

    return losses, accuracies

# parameter scan for ground truth
def parameter_scan_ground_truth(n, m, p, d, s, device, K, linear=False, reps = 1):
    """
    Performs a parameter scan using evaluate_ground_truth, and returns results in a format
    compatible with visualization functions like heatmaps.

    Parameters:
    - n, m, p, d, s, device, K, reps: Same as evaluate_ground_truth
    - linear (bool): Whether to perform a linear scan (synchronized index) or full combination scan.

    Returns:
    - list: A list of dictionaries containing parameters and results.
    """

    # Convert scalar values into lists for iteration
    param_dict = {'n': n, 'm': m, 'p': p, 'd': d, 's': s, 'K': K}
    param_dict = {
        k: list(v) if isinstance(v, np.ndarray) else
           [float(x) if isinstance(x, (np.float32, np.float64)) else int(x) if isinstance(x, np.integer) else x for x in v] 
           if isinstance(v, list) else
           float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, np.integer) else v
        for k, v in param_dict.items()
    }

    # Filter out non-list parameters
    list_params = [v for v in param_dict.values() if isinstance(v, list)]

    # Condition for activating linear scan
    if len(list_params) <= 1:  
        stop = True  # 0 or 1 list → always True
    else:
        stop = all(len(v) == len(list_params[0]) for v in list_params)  # Vérify if all lists have the same length

    for key, value in param_dict.items():
        if not isinstance(value, (list, tuple)):
            param_dict[key] = [value]  # Wrap single values in a list

    results = []

    if linear and stop:
        # Linear scan (synchronized index)
        for i in tqdm(range(len(list_params[0])), desc="Training Progress"):
            params = {k: v[i] if len(v) > 1 else v[0] for k, v in param_dict.items()}
            gt_loss, gt_accuracy = evaluate_ground_truth(**params, device=device, reps = reps) 
            results.append({'params': params, 'results': {'gt_loss': gt_loss, 'gt_accuracy': gt_accuracy}})

    else:
        # Full combination scan
        param_combinations = list(itertools.product(*param_dict.values()))
        for params in tqdm(param_combinations, desc="Training Progress"):
            param_set = dict(zip(param_dict.keys(), params))
            gt_loss, gt_accuracy = evaluate_ground_truth(**param_set, device=device, reps = reps)
            results.append({'params': param_set, 'results': {'gt_loss': gt_loss, 'gt_accuracy': gt_accuracy}})

    return results

