# all imports
import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # type: ignore
import shutil
import time
import os
import subprocess
import webbrowser
import itertools
from generation_data import *
import pickle
from scipy.stats import spearmanr
import numpy as np


############################################################
# This file defines the full pipeline to run matrix factorization experiments 
# using synthetic comparison data (triplets).
#
# Structure and responsibilities:
# - parameter_scan: launches multiple experiments over a hyperparameter grid.
# - run_experiment: runs one experiment (generate data, train, evaluate, log metrics).
# - Model definition: MatrixFactorization class implements the BTL-based embedding model.
# - Dataset management: BTLPreferenceDataset and triplet sampling functions.
# - Evaluation: accuracy, reconstruction error, correlations, etc.
# - Utility functions: save intermediate results, launch TensorBoard, etc.
#
# The results include detailed metrics (train loss, GT accuracy, correlation, etc.)
# and can be used for later visualization.
############################################################

"""
The next function launches a grid search (or linear sweep) over multiple hyperparameter configurations
for matrix factorization experiments using comparison data.

Each parameter can be either:
- a scalar (used for all experiments), or
- a list (to scan different values).

Two scan modes are supported:
- full grid search (all combinations),
- linear scan (synchronized values across lists of equal length).

Each experiment returns a dictionary with:
- 'params': the exact hyperparameters used,
- 'results': a dictionary of performance metrics, including:
    - reconstruction_errors: list of global reconstruction errors (one per repetition),
    - log_likelihoods: list of test log-likelihoods,
    - accuracy: list of prediction accuracies,
    - gt_log_likelihoods: list of ground truth log-likelihoods,
    - gt_accuracy: list of ground truth accuracies,
    - train_losses: list of per-epoch training losses per repetition,
    - val_losses: list of per-epoch validation losses per repetition.

Example structure:
[
    {
        'params': {
            'n': 100, 'm': 200, 'd': 20, 'p': 0.5,
            'lr': 0.001, 'weight_decay': 0.01,
            'num_epochs': 50, 'reps': 3
        },
        'results': {
            'reconstruction_errors': [...],
            'log_likelihoods': [...],
            'accuracy': [...],
            'train_losses': [[...], [...], [...]],
            ...
        }
    },
    ...
]
"""
def parameter_scan(n=1000, m=1000, d=2, p=0.5, s=1.0, device='cpu', 
                   lr=1e-3, weight_decay=1e-5, num_epochs=30, reps=1, strategy="random",
                   open_browser=False, linear=False, K=1, d1=None, 
                   save_path=None, save_every=None, popularity_method="zipf", 
                   alpha=1.5, soft_label=False, generation="base"):
    """
    Launches multiple matrix factorization experiments across different hyperparameter configurations.

    Parameters:
    - n (int or list): Number of users in the preference matrix.
    - m (int or list): Number of items in the preference matrix.
    - d (int or list): Embedding dimension for the learned latent space.
    - p (float or list): Proportion of observed comparisons (fraction of n * m) to generate.
    - s (float or list): Scaling factor applied to the score differences in the BTL model.
    - device (str): Device used for training ('cpu' or 'cuda').
    - lr (float or list): Learning rate for the optimizer (Adam).
    - weight_decay (float or list): L2 regularization coefficient applied during training.
    - num_epochs (int or list): Number of training epochs per experiment.
    - reps (int or list): Number of repeated runs per configuration (with different random seeds).
    - strategy (str): Strategy used to sample triplets (e.g., "random", "margin", "top_k", etc.).
    - open_browser (bool): If True, launches TensorBoard in the web browser at the end of training.
    - linear (bool): If True, performs a linear scan (i.e., uses corresponding values across lists).
                     If False, performs a full Cartesian product over all parameter combinations.
    - K (int): Number of preference labels sampled per triplet (for repeated comparisons).
    - d1 (int or None): Optional secondary dimension for specific model variants (not used if None).
    - save_path (str or None): If set, path to save intermediate/final results (as pickle).
    - save_every (int or None): If set, saves the results every `save_every` experiments.
    - popularity_method (str): Method for popularity-based sampling ("zipf", "uniform", or "exponential").
    - alpha (float): Shape parameter for skewed distributions in some triplet sampling methods.
    - soft_label (bool): If True, uses soft labels (expected value of Bernoulli draws) instead of hard labels.
    - generation (str): Method used to generate the ground-truth matrix X (e.g., "base", "svd", "clustered", etc.).

    Returns:
    - list of dict: Each element contains:
        - 'params': the parameter configuration used,
        - 'results': a dictionary of training/validation/test/reconstruction metrics.
    """


    # === Step 1: Collect all parameters into a single dictionary
    param_dict = {'n': n, 'm': m, 'd': d, 'p': p, 'lr': lr, 
                  'weight_decay': weight_decay, 'num_epochs': num_epochs,
                  'reps': reps, 's': s, 'K': K, 'd1': d1, 'strategy': strategy,
                  'popularity_method': popularity_method, 'alpha': alpha, 
                  'soft_label': soft_label, 'generation': generation}
    
    # === Step 2: Normalize all parameters into lists if needed, converting NumPy types to native Python types
    param_dict = {
        k: list(v) if isinstance(v, np.ndarray) else
           [float(x) if isinstance(x, (np.float32, np.float64)) else int(x) if isinstance(x, np.integer) else x for x in v] 
           if isinstance(v, list) else
           float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, np.integer) else v
        for k, v in param_dict.items()
    }

    # === Step 3: Collect only the parameters that are lists
    list_params = [v for v in param_dict.values() if isinstance(v, list)]

    # === Step 4: Check whether a synchronized linear scan is possible (all lists of the same length)
    if len(list_params) <= 1:
        stop = True  # Only 0 or 1 list → linear scan is allowed
    else:
        stop = all(len(v) == len(list_params[0]) for v in list_params)  # All lists must have same length

    # === Step 5: Wrap scalar values in lists to unify iteration later
    for key, value in param_dict.items():
        if not isinstance(value, (list, tuple)):
            param_dict[key] = [value]

    # === Step 6: If saving path is given and already exists, clear previous results
    if save_path and os.path.exists(save_path):
        print(f"🧹 Removing existing file at {save_path}")
        os.remove(save_path)

    # === CASE 1: Full grid search over all parameter combinations
    if not linear:
        param_combinations = list(itertools.product(*param_dict.values()))
        all_results = []

        for experiment_index, params in enumerate(param_combinations):
            param_set = dict(zip(param_dict.keys(), params))
            print(f"\nRunning experiment with parameters: {param_set}")

            # Run a single experiment
            results = run_experiment(
                n=param_set['n'], m=param_set['m'], d=param_set['d'], p=param_set['p'], 
                s=param_set['s'], device=device, lr=param_set['lr'], 
                weight_decay=param_set['weight_decay'], reps=param_set['reps'], num_epochs=param_set['num_epochs'], 
                open_browser=open_browser, K=param_set['K'], d1=param_set['d1'], strategy=param_set['strategy'],
                popularity_method=param_set['popularity_method'], alpha=param_set['alpha'], soft_label=param_set['soft_label'], generation=param_set['generation']
            )
            all_results.append({'params': param_set, 'results': results})

            # Periodic save: save every N experiments to avoid losing progress
            if save_path and save_every and (len(all_results) >= save_every):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    with open(save_path, 'rb') as f:
                        previous_results = pickle.load(f)
                else:
                    previous_results = []
                previous_results.extend(all_results)
                with open(save_path, 'wb') as f:
                    pickle.dump(previous_results, f)
                print(f"✅ Saved {len(all_results)} new experiments to {save_path}")
                all_results = []  # Clear to free memory

        # Final save after all combinations
        if save_path and len(all_results) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    previous_results = pickle.load(f)
            else:
                previous_results = []
            previous_results.extend(all_results)
            with open(save_path, 'wb') as f:
                pickle.dump(previous_results, f)
            print(f"✅ Saved {len(all_results)} new experiments to {save_path}")
            all_results = []  # Clear memory

        return all_results

    # === CASE 2: Linear scan mode (synchronized lists)
    elif linear and stop:
        all_results = []

        for i in range(len(list_params[0])):
            # Build i-th parameter configuration from synchronized lists
            params = {k: v[i] if len(v) > 1 else v[0] for k, v in param_dict.items()}
            print(f"\nRunning experiment with parameters: {params}")

            # Run a single experiment
            results = run_experiment(
                n=params['n'], m=params['m'], d=params['d'], p=params['p'], 
                s=params['s'], device=device, lr=params['lr'], 
                weight_decay=params['weight_decay'], reps=params['reps'], num_epochs=params['num_epochs'], 
                open_browser=open_browser, K=params['K'], d1=params['d1'], strategy=params['strategy'],
                popularity_method=params['popularity_method'], alpha=params['alpha'], soft_label=params['soft_label'], generation=params['generation']
            )
            all_results.append({'params': params, 'results': results})

            # Periodic save
            if save_path and save_every and (len(all_results) >= save_every):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    with open(save_path, 'rb') as f:
                        previous_results = pickle.load(f)
                else:
                    previous_results = []
                previous_results.extend(all_results)
                with open(save_path, 'wb') as f:
                    pickle.dump(previous_results, f)
                print(f"✅ Saved {len(all_results)} new experiments to {save_path}")
                all_results = []  # Clear memory

        # Final save
        if save_path and len(all_results) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    previous_results = pickle.load(f)
            else:
                previous_results = []
            previous_results.extend(all_results)
            with open(save_path, 'wb') as f:
                pickle.dump(previous_results, f)
            print(f"✅ Saved {len(all_results)} new experiments to {save_path}")
            all_results = []  # Clear memory

        return all_results

    # === CASE 3: Invalid linear configuration (mismatched list lengths)
    else:
        raise ValueError("The linear scan is not possible because the parameters are not synchronized.")


def print_return_structure_types(obj, prefix="root"):
    """
    Recursively prints the type structure of a nested object. (for debugging purposes)
    
    - If the object is a dictionary, it traverses all key-value pairs recursively.
    - If the object is a list or tuple, it reports the type of its elements:
        • If all elements share the same type, the type is printed.
        • If types differ, it reports 'mixed'.
        • If empty, reports '[empty]'.
    - If the object is a PyTorch tensor, it explicitly reports 'torch.Tensor'.
    - Otherwise, it prints the Python type name.
    
    Parameters:
    - obj: the input object (can be a dict, list, tuple, tensor, or scalar).
    - prefix (str): string prefix indicating the current position in the nested structure.
    """

    # === Case 1: dictionary — recursively explore its contents
    if isinstance(obj, dict):
        for k, v in obj.items():
            print_return_structure_types(v, f"{prefix}.{k}")

    # === Case 2: list or tuple — check the types of all elements
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            # Special case: empty list or tuple
            print(f"{prefix}: {type(obj).__name__}[empty]")
        else:
            # Collect all types of elements
            inner_types = set(type(el).__name__ for el in obj)
            if len(inner_types) == 1:
                # All elements have the same type
                inner_type = next(iter(inner_types))
                print(f"{prefix}: {type(obj).__name__}[{inner_type}]")
            else:
                # Elements have mixed types
                print(f"{prefix}: {type(obj).__name__}[mixed]")

    # === Case 3: PyTorch tensor — report explicitly
    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}: torch.Tensor")

    # === Case 4: any other type — report the type name
    else:
        print(f"{prefix}: {type(obj).__name__}")


# Main function to run a single experiment configuration, possibly repeated multiple times
def run_experiment(n, m, d, p, s, device, lr, weight_decay, reps=5, num_epochs=100, open_browser=False, K=1, d1=None, strategy="random", popularity_method="zipf", alpha=1.5, soft_label=False, generation="base"):
    """
    Runs multiple repetitions of a matrix factorization experiment using triplet comparison data.
    Preference labels follow a Bradley-Terry-Luce (BTL) model.
    All metrics are aggregated and returned.

    Parameters:
    - n (int): Number of users.
    - m (int): Number of items.
    - d (int): Latent embedding dimension.
    - p (float): Proportion of observed user-item comparisons.
    - s (float): Scaling factor applied to preference scores.
    - device (str): "cpu" or "cuda".
    - lr (float): Learning rate.
    - weight_decay (float): L2 regularization strength.
    - reps (int): Number of independent repetitions.
    - num_epochs (int): Number of training epochs.
    - open_browser (bool): If True, open TensorBoard after training.
    - K (int): Number of labels per triplet (repeated votes).
    - d1 (int or None): Optional secondary latent dimension (not used here).
    - strategy (str): Triplet sampling strategy ("random", "margin", etc.).
    - popularity_method (str): Distribution type for popularity-based sampling.
    - alpha (float): Skewness parameter for Zipf/exponential distributions.
    - soft_label (bool): If True, use soft labels for training.
    - generation (str): Method used to generate the ground-truth matrix X.

    Returns:
    - dict: Aggregated metrics across all repetitions.
    """

    # === Initialize metric containers ===
    alpha_vals, norm_X_vals, norm_ratio_vals = [], [], []
    recs_scaled, pearson_means, pearson_stds = [], [], []
    spearman_means, spearman_stds, svd_errors = [], [], []
    reconstruction_errors, log_likelihoods, accuracy = [], [], []
    gt_accuracy, gt_log_likelihoods, train_losses, val_losses = [], [], [], []
    slopes_matrix, correlations_matrix, spearman_scores_matrix = [], [], []
    reconstruction_error_scaled_per_row_matrix = []
    alpha_per_row_matrix = []   
    sampled_UVT_rows_matrix, sampled_X_rows_matrix = [], []

    # === Repeat the full experiment 'reps' times ===
    for rep in range(reps):
        if d1 is None:
            d1 = d  # fallback to main latent dimension if not specified

        # === Step 1: Generate ground-truth preference matrix X (n x m) ===
        X = generate_X(n, m, d, device, generation=generation)

        # === Step 2: Generate triplet dataset and split into train/val/test sets ===
        num_triplets = int(n * m * p / 2)
        train_loader, val_loader, test_loader = split_dataset_from_triplets(
            X, num_triplets, scale=s, K=K, strategy=strategy,
            popularity_method=popularity_method, alpha=alpha, soft_label=soft_label
        )

        # === Step 3: Initialize model and optimizer ===
        model = MatrixFactorization(n, m, d).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # === Step 4: Train the model and collect losses ===
        is_last = (rep == reps - 1)
        t_losses, v_losses = train_model(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, is_last=is_last, open_browser=open_browser
        )
        train_losses.append(t_losses)
        val_losses.append(v_losses)

        # === Step 5: Evaluate model accuracy and log-likelihood ===
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        accuracy.append(test_acc)
        log_likelihoods.append(-test_loss)

        # === Step 6: Compute scaled reconstruction error ===
        rec_error = compute_reconstruction_error(model, X, s)
        reconstruction_errors.append(rec_error)

        # === Step 6b: Compute structural alignment metrics (alpha, Pearson/Spearman correlations, etc.) ===
        alpha_val, norm_X_val, norm_ratio_val, rec_scaled, pearson_mean, pearson_std, spearman_mean, spearman_std, svd_err, slopes, correlations, spearman_scores, reconstruction_error_scaled_per_row, alpha_per_row = compute_alpha_and_norm_ratios(model, X)

        # === Step 6c: Sample 2 random rows for visual inspection (X and UVᵀ) ===
        with torch.no_grad():
            UVT_full = torch.matmul(model.U, model.V.t())
            rand_indices = torch.randperm(X.shape[0])[:2]
            sampled_X_rows = X[rand_indices].cpu().numpy()
            sampled_UVT_rows = UVT_full[rand_indices].cpu().numpy()

        # === Step 6d: Store all computed metrics for this repetition ===
        alpha_vals.append(alpha_val)
        norm_X_vals.append(norm_X_val)
        norm_ratio_vals.append(norm_ratio_val)
        recs_scaled.append(rec_scaled)
        pearson_means.append(pearson_mean)
        pearson_stds.append(pearson_std)
        spearman_means.append(spearman_mean)
        spearman_stds.append(spearman_std)
        svd_errors.append(svd_err)
        slopes_matrix.append(slopes)
        correlations_matrix.append(correlations)
        spearman_scores_matrix.append(spearman_scores)

        # === Step 7: Evaluate using ground-truth model (no learning) ===
        gt_loss, gt_acc = compute_ground_truth_metrics(test_loader, X, device)
        gt_log_likelihoods.append(-gt_loss)
        gt_accuracy.append(gt_acc)

        # === Step 8: Store per-row reconstruction error and per-row alpha values ===
        reconstruction_error_scaled_per_row_matrix.append(reconstruction_error_scaled_per_row)
        alpha_per_row_matrix.append(alpha_per_row)
        sampled_UVT_rows_matrix.append(sampled_UVT_rows)
        sampled_X_rows_matrix.append(sampled_X_rows)

    # === Collect all metrics into a structured result dictionary ===
    raw_result = {
        "reconstruction_errors": reconstruction_errors,
        "log_likelihoods": log_likelihoods,
        "accuracy": accuracy,
        "gt_log_likelihoods": gt_log_likelihoods,
        "gt_accuracy": gt_accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "alpha": alpha_vals,
        "norm_X": norm_X_vals,
        "norm_ratio": norm_ratio_vals,
        "reconstruction_error_scaled": recs_scaled,
        "pearson_corr": pearson_means,
        "pearson_std": pearson_stds,
        "spearman_corr": spearman_means,
        "spearman_std": spearman_stds,
        "svd_error_scaled": svd_errors,
        "slopes": slopes_matrix,
        "pearson_corr_matrix": correlations_matrix,
        "spearman_corr_matrix": spearman_scores_matrix,
        "reconstruction_error_scaled_per_row": reconstruction_error_scaled_per_row_matrix,
        "alpha_per_row": alpha_per_row_matrix,
        "sampled_UVT_rows": sampled_UVT_rows_matrix,
        "sampled_X_rows": sampled_X_rows_matrix
    }
    # === Print the structure of the return object for debugging (uncomment if needed) ===
    # print("🔍 Types in return object:")
    # print_return_structure_types(raw_result)

    # === Return all computed metrics for external analysis or visualization ===
    return raw_result




############################################
# Dataset definition and triplet generation logic:
# This section defines how triplet-based preference datasets are created and used,
# including:
# - The BTLPreferenceDataset class for (u, i, j, label) tuples
# - The get_triplets_from_X function to generate triplets using a sampling strategy
# - The generate_X function to produce the ground-truth matrix X
# - The split_dataset_from_triplets function to split triplets into train/val/test loaders
############################################

class BTLPreferenceDataset(Dataset):
    """
    PyTorch Dataset for generating preference labels under the Bradley-Terry-Luce (BTL) model.
    
    Each datapoint corresponds to a triplet (u, i, j), representing a comparison between items i and j by user u.
    The label is sampled according to the BTL probability:
        P(u prefers i over j) = sigmoid(scale * (X[u, i] - X[u, j]))
    
    Supports both hard (binary) and soft labels, and multiple repetitions (K) per triplet.
    """

    def __init__(self, triplets, X, scale=1.0, K=1, soft_label=False, train=False):
        """
        Initializes the dataset.

        Parameters:
        - triplets (list of tuples): List of (u, i, j) triplet indices.
        - X (torch.Tensor): Ground-truth preference matrix of size (n_users x n_items).
        - scale (float): Scaling factor for the score differences in the BTL model.
        - K (int): Number of repetitions (votes) per triplet.
        - soft_label (bool): If True, use soft (expected) label instead of binary samples.
        - train (bool): If True, use soft labels only during training.
        """
        self.X = X
        self.scale = scale
        self.soft_label = soft_label
        self.data = self._generate_labels(triplets, K, train=train)

    def _generate_labels(self, triplets, K, train=False):
        """
        Generates labeled data from triplets using the BTL model.

        For each (u, i, j), compute the preference probability and sample labels.

        - If soft_label is True and training mode is enabled:
            • Use the expected value (average over K Bernoulli samples).
        - Otherwise:
            • Draw K independent binary labels from the Bernoulli distribution.

        Returns:
        - data (list of tuples): Each entry is (u, i, j, label)
        """
        data = []
        for (u, i, j) in triplets:
            score = torch.sigmoid(self.scale * (self.X[u, i] - self.X[u, j]))
            if self.soft_label and train:
                # Sample K binary values and average them → soft label
                label = torch.mean(torch.bernoulli(score.expand(K))).item()
                data.append((u, i, j, label))
            else:
                # Sample K independent binary labels
                for _ in range(K):
                    label = torch.bernoulli(score).item()
                    data.append((u, i, j, label))
        return data

    def __len__(self):
        """
        Returns the number of (u, i, j, label) examples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the i-th (u, i, j, label) sample from the dataset.
        """
        return self.data[idx]

def get_triplets_from_X(X, num_triplets, strategy="random", exclude=None,
                        popularity_method="zipf", alpha=1.5, n_clusters=10):
    """
    Generates (u, i, j) triplets from a ground-truth preference matrix X, using a specified sampling strategy.

    Parameters:
    - X (Tensor): The ground-truth user-item matrix of size (n_users x n_items).
    - num_triplets (int): Total number of triplets to sample.
    - strategy (str): Sampling strategy to use. Supported options include:
        * "random"       → fully random triplets
        * "proximity"    → select items with large/small scores
        * "margin"       → select item pairs with similar scores (Close-Call)
        * "variance"     → sample from high-variance items
        * "popularity"   → skew sampling by popularity distribution (Min-Max)
        * "top_k"        → restrict sampling to top-k ranked items (top_10%)
        * "cluster"      → select from different item clusters
        * "user_similarity" → exploit user similarity
        * "svd"          → use latent projections to bias sampling
    - exclude (set): Optional set of triplets to avoid (already used triplets).
    - popularity_method (str): Method for popularity-based sampling: "zipf", "uniform", or "exponential".
    - alpha (float): Shape parameter for skewness in Zipf or exponential distributions.
    - n_clusters (int): Number of clusters used when strategy is "cluster".

    Returns:
    - set: A set of unique triplets (u, i, j) sampled according to the selected strategy.
    """

    # Initialize exclusion set if None
    exclude = exclude or set()

    # === Dispatch to the appropriate triplet selection strategy ===
    if strategy == "random":
        candidates = choose_items_random(X, num_triplets=num_triplets, exclude=exclude)
    elif strategy == "proximity":
        candidates = choose_items_by_proximity(X, num_triplets, exclude)
    elif strategy == "margin":
        candidates = choose_items_by_margin(X, num_triplets, exclude)
    elif strategy == "variance":
        candidates = choose_items_by_variance(X, num_triplets, exclude)
    elif strategy == "popularity":
        candidates = choose_items_by_popularity(
            X, num_triplets, exclude, method=popularity_method, alpha=alpha
        )
    elif strategy == "top_k":
        candidates = choose_items_top_k(X, num_triplets, exclude)
    elif strategy == "cluster":
        candidates = choose_items_cluster_based(X, num_triplets, exclude, n_clusters=n_clusters)
    elif strategy == "user_similarity":
        candidates = choose_items_by_user_similarity(X, num_triplets, exclude)
    elif strategy == "svd":
        candidates = choose_items_by_svd_projection(X, num_triplets, exclude)
    else:
        raise ValueError(f"Unknown triplet sampling strategy: {strategy}")

    # Return the selected triplets as a set (to ensure uniqueness)
    return set(candidates)

def generate_X(n, m, d, device, generation="base", **kwargs):
    """
    Wrapper function that generates a user-item preference matrix X using a specified generation scheme.
    
    Parameters:
    - n (int): Number of users (rows).
    - m (int): Number of items (columns).
    - d (int): Latent dimension.
    - device (str): Target device ("cpu" or "cuda") for the resulting tensor.
    - generation (str): Name of the generation method to use. Supported values:
        • "base", "low_rank", "structured", "svd", "correlated", "graph",
        • "social", "temporal", "hierarchical", "gmm", "clustered"
    - kwargs: Additional keyword arguments required for specific generators (e.g., `rank`).

    Returns:
    - X (Tensor): A generated (n x m) matrix representing latent preferences.
    """

    # === Basic orthogonal embedding-based matrix ===
    if generation == "base":
        return generate_embeddings(n, m, d, device=device)

    # === Low-rank matrix with explicit diagonal scaling (U diag(S) Vᵀ) ===
    elif generation == "low_rank":
        U, V, S = generate_low_rank_matrix(n, m, d, rank=kwargs.get("rank", d), device=device)
        return torch.matmul(torch.matmul(U, torch.diag(S)), V.t())

    # === Structured clustering in latent space ===
    elif generation == "structured":
        U, V = generate_structured_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Decomposition-based embeddings using SVD on a noisy matrix ===
    elif generation == "svd":
        U, V = generate_svd_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Introduce linear correlations across latent dimensions ===
    elif generation == "correlated":
        U, V = generate_correlated_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Graph-based structure using social graph topology ===
    elif generation == "graph":
        U, V = generate_graph_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Social influence-based embeddings: users near their friends ===
    elif generation == "social":
        U, V = generate_social_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Embeddings that evolve over time (drifting preferences) ===
    elif generation == "temporal":
        U, V = generate_temporal_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Hierarchical embeddings: users grouped into latent classes ===
    elif generation == "hierarchical":
        U, V = generate_hierarchical_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Embeddings sampled from a Gaussian Mixture Model ===
    elif generation == "gmm":
        U, V = generate_gmm_embeddings(n, m, d, device=device)
        return torch.matmul(U, V.t())

    # === Embeddings where item vectors are softly aligned to clusters ===
    elif generation == "clustered":
        return generate_clustered_matrix_from_embeddings(n, m, d, device=device)

    # === Unknown generation keyword → raise an error ===
    else:
        raise ValueError(f"Unknown generation method: {generation}")


def split_dataset_from_triplets(X, num_triplets, scale=1.0, K=1,
                                 train_ratio=0.8, val_ratio=0.1,
                                 batch_size=64, strategy="random",
                                 popularity_method="zipf", alpha=1.5, soft_label=False):
    """
    Generates a dataset of (u, i, j) triplets from a ground-truth matrix using a given strategy,
    then splits them into training, validation, and test DataLoaders with BTL-style labels.

    Parameters:
    - X (Tensor): Ground-truth user-item matrix of shape (n x m).
    - num_triplets (int): Total number of triplets to generate.
    - scale (float): Scaling factor applied to BTL sigmoid probabilities.
    - K (int): Number of label repetitions per triplet (Bernoulli draws).
    - train_ratio (float): Proportion of triplets assigned to the training set.
    - val_ratio (float): Proportion of triplets assigned to the validation set.
    - batch_size (int): Batch size used for each DataLoader.
    - strategy (str): Triplet sampling strategy to use (e.g. "random", "proximity", "popularity").
    - popularity_method (str): Distribution type for popularity-based sampling.
    - alpha (float): Shape/skewness parameter for Zipf/exponential distributions.
    - soft_label (bool): Whether to use soft labels (expected value over K votes).

    Returns:
    - train_loader (DataLoader): DataLoader over the training triplets.
    - val_loader (DataLoader): DataLoader over the validation triplets.
    - test_loader (DataLoader): DataLoader over the test triplets.
    """

    n, m = X.shape  # Get matrix dimensions

    # === Step 1: Generate triplets using the given strategy ===
    triplets = list(get_triplets_from_X(
        X, num_triplets, strategy=strategy,
        popularity_method=popularity_method, alpha=alpha
    ))
    if len(triplets) < num_triplets:
        print(f"⚠️ Only {len(triplets)} triplets generated for strategy: {strategy} (target={num_triplets})")

    # === Step 2: Split triplets into train / val / test sets ===
    total = len(triplets)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    # Use reproducible random split
    train_split, val_split, test_split = torch.utils.data.random_split(
        triplets, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # === Step 3: Extract triplet lists from the splits ===
    train_triplets = [triplets[i] for i in train_split.indices]
    val_triplets = [triplets[i] for i in val_split.indices]
    test_triplets = [triplets[i] for i in test_split.indices]

    # === Step 4: Ensure minimum size for the test set (at least MIN_TEST_POINTS) ===
    MIN_TEST_POINTS = 500
    if len(test_triplets) * K < MIN_TEST_POINTS:
        seen_triplets = set(train_triplets + val_triplets + test_triplets)
        needed = (MIN_TEST_POINTS + K - 1) // K - len(test_triplets)
        extra = get_triplets_from_X(
            X, needed, strategy=strategy,
            popularity_method=popularity_method, alpha=alpha,
            exclude=seen_triplets
        )
        test_triplets += list(extra)

    # === Step 5: Wrap triplets in Dataset objects (BTL formulation) ===
    train_dataset = BTLPreferenceDataset(train_triplets, X, scale=scale, K=K, soft_label=soft_label, train=True)
    val_dataset = BTLPreferenceDataset(val_triplets, X, scale=scale, K=K, soft_label=soft_label)
    test_dataset = BTLPreferenceDataset(test_triplets, X, scale=scale, K=K, soft_label=soft_label)

    # === Step 6: Convert datasets into PyTorch DataLoaders ===
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
# Training and Evaluation Utilities
# 
# This section contains core utility functions for:
# - Training the matrix factorization model (train_model)
# - Evaluating model performance on test data (evaluate_model)
# - Computing reconstruction and alignment metrics (compute_reconstruction_error, compute_alpha_and_norm_ratios)
# - Evaluating performance using the ground-truth matrix directly (compute_ground_truth_metrics)
# - Launching TensorBoard for experiment monitoring (start_tensorboard)
############################################


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, is_last=False, open_browser=False):
    """
    Trains a matrix factorization model using binary cross-entropy loss between predicted preferences and true labels.

    Parameters:
    - model (torch.nn.Module): The matrix factorization model to be trained.
    - train_loader (DataLoader): Batches of training (u, i, j, label) samples.
    - val_loader (DataLoader): Batches of validation samples.
    - optimizer (torch.optim.Optimizer): Optimizer used for gradient descent (e.g., Adam).
    - device (torch.device): Device on which training will be performed ("cpu" or "cuda").
    - num_epochs (int): Number of training epochs to perform.
    - is_last (bool): Whether this is the final experiment (can be used to trigger special logging).
    - open_browser (bool): If True and TensorBoard is enabled, open it in a web browser.

    Returns:
    - (list, list): Two lists containing training and validation loss values for each epoch.
    """
    
    # === Optional: TensorBoard logging (disabled by default, change "if False" to "if True" to enable) ===
    if False:
        log_dir = 'runs/matrix_factorization'
        start_tensorboard(log_dir=log_dir, open_browser=open_browser)
        writer = SummaryWriter(log_dir=log_dir)
    
    train_losses = []
    val_losses = []

    # === Training loop over epochs ===
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        total_loss = 0
        model.train()  # Enable training mode (activates dropout, batchnorm, etc.)

        # === Training phase ===
        for batch in train_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Transfer batch to device
            optimizer.zero_grad()  # Clear previous gradients
            pred = model(u, i, j)  # Forward pass
            loss = F.binary_cross_entropy(pred, z.float())  # Compute binary cross-entropy loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            total_loss += loss.item()  # Accumulate total loss

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # === Validation phase ===
        val_loss = 0
        model.eval()  # Evaluation mode (no dropout, etc.)
        with torch.no_grad():  # Disable gradient computation during evaluation
            for batch in val_loader:
                u, i, j, z = [x.to(device) for x in batch]
                pred = model(u, i, j)
                loss = F.binary_cross_entropy(pred, z.float())
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # === Optional: log losses to TensorBoard ===
        if False:
            writer.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, epoch)

    # === Optional: close the TensorBoard writer ===
    if False:
        writer.close()

    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """
    Evaluates a trained matrix factorization model on a held-out test set.

    Parameters:
    - model (torch.nn.Module): The trained model to be evaluated.
    - test_loader (DataLoader): DataLoader providing batches of (u, i, j, label) for evaluation.
    - device (torch.device): Device on which evaluation will be performed ("cpu" or "cuda").

    Returns:
    - (float, float): Tuple containing:
        • average binary cross-entropy loss over the test set,
        • accuracy (i.e., percentage of correctly predicted pairwise preferences).
    """

    model.eval()  # Switch model to evaluation mode (disables dropout, etc.)
    test_loss, correct, total = 0, 0, 0  # Initialize metrics

    # === Evaluation loop (no gradients) ===
    with torch.no_grad():
        for batch in test_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Transfer batch to device

            # === Forward pass to get predicted preference probabilities ===
            pred = model(u, i, j)

            # === Compute binary cross-entropy loss against true labels ===
            loss = F.binary_cross_entropy(pred, z.float())
            test_loss += loss.item()

            # === Binarize predictions at threshold 0.5 ===
            pred = (pred > 0.5).float()

            # === Count correct predictions ===
            correct += (pred == z).sum().item()
            total += len(z)

    # === Compute accuracy as proportion of correct predictions ===
    accuracy = correct / total if total > 0 else 0.0

    return test_loss / len(test_loader), accuracy


# Reconstruction error
def compute_reconstruction_error(model, X, s):
    """
    Computes the reconstruction error between the predicted user-item matrix (UVᵀ)
    and the ground-truth matrix X, using Frobenius norm.

    Parameters:
    - model (torch.nn.Module): Trained matrix factorization model with U and V embeddings.
    - X (torch.Tensor): Ground-truth user-item matrix (shape: n x m).
    - s (float): Scaling factor applied to the ground-truth matrix.

    Returns:
    - float: Normalized reconstruction error (Frobenius norm ratio).
    """

    # === Step 1: Reconstruct the matrix UVᵀ from learned embeddings ===
    user_item_matrix = torch.mm(model.U, model.V.t())

    # === Step 2: Center the reconstructed matrix by subtracting the column mean ===
    user_item_matrix -= torch.mean(user_item_matrix, dim=0, keepdim=True)

    # === Step 3: Compute the Frobenius norm of the scaled ground-truth matrix ===
    frobenius_norm = torch.norm(s * X, p="fro")

    # === Step 4: Compute Frobenius norm of the reconstruction error ===
    reconstruction_error = torch.norm(user_item_matrix - s * X, p="fro")

    # === Step 5: Normalize error by the Frobenius norm of ground truth ===
    normalized_error = reconstruction_error / frobenius_norm

    # === Step 6: Return scalar value as float ===
    return normalized_error.item()


def compute_alpha_and_norm_ratios(model, X_init):
    """
    Computes multiple alignment and structural similarity metrics between the predicted matrix UVᵀ
    and the ground-truth matrix X.

    Returns:
    - alpha (float): Optimal scalar to align UVᵀ with X (least squares).
    - norm_X (float): Frobenius norm of X.
    - norm_ratio (float): Frobenius norm ratio ||UVᵀ|| / ||X||.
    - reconstruction_error_scaled (float): Normalized Frobenius distance between α·UVᵀ and X.
    - pearson_mean (float): Mean Pearson correlation across rows.
    - pearson_std (float): Standard deviation of Pearson correlations.
    - spearman_mean (float): Mean Spearman correlation across rows.
    - spearman_std (float): Standard deviation of Spearman correlations.
    - svd_error_scaled (float): Normalized difference in singular values.
    - slopes (list): Linear regression slopes between X[i] and UVᵀ[i].
    - correlations (list): Row-wise Pearson correlation values.
    - spearman_scores (list): Row-wise Spearman correlation values.
    - reconstruction_error_scaled_per_row (float): Frobenius distance using per-row αᵢ.
    - alpha_per_row (list): List of optimal αᵢ values per row.
    """

    with torch.no_grad():
        # === Compute predicted matrix UVᵀ ===
        UVT = torch.matmul(model.U, model.V.t())

        # === Center the predicted matrix and the ground truth matrix row-wise ===
        UVT -= torch.mean(UVT, dim=1, keepdim=True)
        X = X_init.clone()
        X -= torch.mean(X, dim=1, keepdim=True)

        # === Compute scalar α that minimizes ||α·UVᵀ - X||_F ===
        dot_product = torch.sum(UVT * X)
        norm_UVT = torch.norm(UVT, p="fro")
        norm_X = torch.norm(X, p="fro")

        alpha = dot_product / (norm_UVT ** 2 + 1e-8)
        norm_ratio = norm_UVT / (norm_X + 1e-8)
        reconstruction_error_scaled = torch.norm(alpha * UVT - X, p="fro") / (norm_X + 1e-8)

        # === Compute row-wise Pearson correlations ===
        X_np = X.cpu().numpy()
        UVT_np = UVT.cpu().numpy()
        n = X_np.shape[0]
        correlations = []
        for i in range(n):
            x_row = X_np[i, :]
            u_row = UVT_np[i, :]
            if np.std(x_row) > 1e-8 and np.std(u_row) > 1e-8:
                corr = np.corrcoef(x_row, u_row)[0, 1]
                correlations.append(corr)
        pearson_mean = float(np.mean(correlations)) if correlations else 0.0

        # === Compute singular value difference between α·UVᵀ and X ===
        try:
            U1, S1, V1 = torch.linalg.svd(X)
            U2, S2, V2 = torch.linalg.svd(UVT)
            k = min(len(S1), len(S2))
            S_diff = alpha * S2[:k] - S1[:k]
            svd_error_scaled = torch.norm(S_diff, p=2) / (torch.norm(S1[:k], p=2) + 1e-8)
        except:
            pearson_mean = 0.0
            svd_error_scaled = 1.0

        # === Compute row-wise Spearman correlations ===
        spearman_scores = []
        for i in range(n):
            x_row = X_np[i, :]
            u_row = UVT_np[i, :]
            if np.std(x_row) > 1e-8 and np.std(u_row) > 1e-8:
                rho, _ = spearmanr(x_row, u_row)
                if not np.isnan(rho):
                    spearman_scores.append(rho)
        spearman_mean = float(np.mean(spearman_scores)) if spearman_scores else 0.0

        # === Standard deviation of correlation values ===
        pearson_std = float(np.std(correlations)) if correlations else 0.0
        spearman_std = float(np.std(spearman_scores)) if spearman_scores else 0.0

        # === Linear regression slopes between each row of X and UVᵀ ===
        slopes = []
        for i in range(n):
            x_row = X_np[i, :]
            u_row = UVT_np[i, :]
            denom = np.dot(x_row, x_row)
            if denom > 1e-8 and np.std(u_row) > 1e-8:
                slope = np.dot(x_row, u_row) / denom
                slopes.append(slope)

        # === Compute per-row optimal scaling αᵢ to align each UVᵀ[i] with X[i] ===
        alpha_per_row = []
        adjusted_rows = []
        X_np = X.cpu().numpy()
        UVT_np = UVT.cpu().numpy()
        n = X_np.shape[0]

        for i in range(n):
            u_row = UVT_np[i]
            x_row = X_np[i]
            denom = np.dot(u_row, u_row)
            alpha_i = np.dot(x_row, u_row) / denom if denom > 1e-8 else 0.0
            alpha_per_row.append(alpha_i)
            adjusted_rows.append(alpha_i * u_row)

        # === Stack adjusted rows to form an adjusted matrix and compute its error ===
        adjusted_matrix = torch.tensor(np.stack(adjusted_rows), dtype=torch.float32)
        reconstruction_error_scaled_per_row = torch.norm(adjusted_matrix - X, p="fro") / (norm_X + 1e-8)

        # === Return all metrics ===
        return (
            alpha.item(),
            norm_X.item(),
            norm_ratio.item(),
            reconstruction_error_scaled.item(),
            pearson_mean,
            pearson_std,
            spearman_mean,
            spearman_std,
            svd_error_scaled.item(),
            slopes,
            correlations,
            spearman_scores,
            reconstruction_error_scaled_per_row.item(),
            alpha_per_row,
        )

# Ground truth metrics
def compute_ground_truth_metrics(test_loader, X, device):
    """
    Computes evaluation metrics using the ground-truth preference matrix X instead of a learned model.

    Parameters:
    - test_loader (DataLoader): Batches of (u, i, j, label) for evaluation.
    - X (torch.Tensor): Ground-truth user-item matrix of shape (n x m).
    - device (torch.device): Device on which the computation will be performed.

    Returns:
    - (float, float): Tuple containing:
        • Mean Squared Error (MSE) between ground-truth probabilities and labels.
        • Accuracy (percentage of correct binary preference predictions).
    """

    gt_loss, gt_correct, gt_total = 0, 0, 0  # Initialize metric accumulators

    # === Evaluate each batch without computing gradients ===
    with torch.no_grad():
        for batch in test_loader:
            u, i, j, z = [x.to(device) for x in batch]  # Move batch to device

            # === Compute preference score difference from ground truth matrix ===
            diff = X[u, i] - X[u, j]  # Higher score → item i is preferred

            # === Convert score difference to probability via sigmoid ===
            gt_prob = torch.sigmoid(diff)

            # === Compute MSE loss between predicted probabilities and true labels ===
            loss = F.mse_loss(gt_prob, z.float())
            gt_loss += loss.item()

            # === Convert diff to hard label (1 if i preferred over j, else 0) ===
            pred = (diff > 0).float()

            # === Count how many predictions match the ground-truth labels ===
            gt_correct += (pred == z).sum().item()
            gt_total += len(z)

    # === Compute average accuracy across all predictions ===
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
# Unit Tests for Ground-Truth Evaluation Functions
# These validate that compute_ground_truth_metrics behaves as expected.
############################################

# Test function to evaluate performance using the ground-truth matrix directly
def evaluate_ground_truth(n, m, p, d, s, device, K, reps=1, strategy="random", popularity_method="zipf", alpha=1.5, soft_label=False, generation="base"):
    """
    Generates a ground-truth matrix, simulates preference data, and computes evaluation metrics
    based purely on the true matrix (without training a model).

    Parameters:
    - n (int): Number of users (rows in the matrix).
    - m (int): Number of items (columns in the matrix).
    - p (float): Proportion of possible user-item pairs used for training.
    - d (int): Latent dimension size for embedding generation.
    - s (float): Scaling factor for sigmoid transformation of scores.
    - device (str): "cpu" or "cuda" for computation.
    - K (int): Number of Bernoulli label draws per (u, i, j) triplet.
    - reps (int): Number of repetitions to average over.
    - strategy (str): Triplet sampling strategy.
    - popularity_method (str): Popularity distribution for certain strategies.
    - alpha (float): Skew parameter for Zipf/exponential distributions.
    - soft_label (bool): Whether to use soft labels (expected Bernoulli mean).
    - generation (str): Method used to generate the matrix X.

    Returns:
    - (list, list): Lists of losses and accuracies over all repetitions.
    """

    losses = []
    accuracies = []

    for _ in range(reps):
        # === Generate ground-truth matrix using the selected method ===
        X = generate_X(n, m, d, device, generation=generation)

        # === Determine number of preference triplets to generate ===
        num_triplets = int(n * m * p / 2)

        # === Create test dataset from triplets using selected strategy ===
        _, _, test_loader = split_dataset_from_triplets(
            X, num_triplets, scale=s, K=K,
            strategy=strategy, popularity_method=popularity_method,
            alpha=alpha, soft_label=soft_label
        )

        # === Compute ground-truth accuracy and loss ===
        gt_loss, gt_acc = compute_ground_truth_metrics(test_loader, X, device)
        losses.append(gt_loss)
        accuracies.append(gt_acc)

    return losses, accuracies

# Parameter sweep for evaluating ground-truth performance without training a model
def parameter_scan_ground_truth(n, m, p, d, s, device, K, linear=False, reps=1, strategy="random", popularity_method="zipf", alpha=1.5, soft_label=False, generation="base"):
    """
    Performs a parameter sweep using evaluate_ground_truth, which simulates preference data and computes
    loss/accuracy from the ground-truth matrix directly (no model training). Results are formatted for plotting.

    Parameters:
    - n, m, p, d, s (int or list): Hyperparameters for matrix and preference generation.
    - device (str): "cpu" or "cuda" for computation.
    - K (int): Number of Bernoulli draws per triplet.
    - linear (bool): If True, synchronize parameter lists and iterate linearly. If False, do full Cartesian scan.
    - reps (int): Number of repetitions for averaging metrics.
    - strategy (str): Triplet sampling strategy.
    - popularity_method (str): Method used for popularity-based strategies.
    - alpha (float): Skew parameter for Zipf/exponential.
    - soft_label (bool): Use soft labels (mean over Bernoulli draws).
    - generation (str): Method for generating matrix X.

    Returns:
    - list[dict]: Each dictionary contains a 'params' dict and a 'results' dict with gt_loss and gt_accuracy.
    """

    # === Normalize input parameters into lists for iteration ===
    param_dict = {
        'n': n, 'm': m, 'p': p, 'd': d, 's': s, 'K': K, 'strategy': strategy,
        'popularity_method': popularity_method, 'alpha': alpha,
        'soft_label': soft_label, 'generation': generation
    }
    param_dict = {
        k: list(v) if isinstance(v, np.ndarray) else
           [float(x) if isinstance(x, (np.float32, np.float64)) else int(x) if isinstance(x, np.integer) else x for x in v] 
           if isinstance(v, list) else
           float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, np.integer) else v
        for k, v in param_dict.items()
    }

    # === Identify all parameters that are lists ===
    list_params = [v for v in param_dict.values() if isinstance(v, list)]

    # === Determine if a linear scan is valid (synchronized lengths) ===
    if len(list_params) <= 1:
        stop = True  # 0 or 1 list: always allowed
    else:
        stop = all(len(v) == len(list_params[0]) for v in list_params)

    # === Ensure every parameter is in list format ===
    for key, value in param_dict.items():
        if not isinstance(value, (list, tuple)):
            param_dict[key] = [value]

    results = []

    # === Case 1: Linear scan (synchronized indexing) ===
    if linear and stop:
        for i in tqdm(range(len(list_params[0])), desc="Training Progress"):
            params = {k: v[i] if len(v) > 1 else v[0] for k, v in param_dict.items()}
            gt_loss, gt_accuracy = evaluate_ground_truth(**params, device=device, reps=reps)
            results.append({'params': params, 'results': {'gt_loss': gt_loss, 'gt_accuracy': gt_accuracy}})

    # === Case 2: Full grid scan (Cartesian product) ===
    else:
        param_combinations = list(itertools.product(*param_dict.values()))
        for params in tqdm(param_combinations, desc="Training Progress"):
            param_set = dict(zip(param_dict.keys(), params))
            gt_loss, gt_accuracy = evaluate_ground_truth(**param_set, device=device, reps=reps)
            results.append({'params': param_set, 'results': {'gt_loss': gt_loss, 'gt_accuracy': gt_accuracy}})

    return results


