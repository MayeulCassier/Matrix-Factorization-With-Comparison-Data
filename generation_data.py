import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch # type: ignore
import networkx as nx
from sklearn.mixture import GaussianMixture
from scipy.stats import ortho_group
import numpy as np
from sklearn.cluster import KMeans

###########################################################
# New triplet selection strategies
###########################################################

# === RANDOM ===
def choose_items_random(X, num_triplets, exclude):
    """Randomly sample i and j."""
    n, m = X.shape
    triplets = set()
    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        i, j = torch.randint(0, m, (2,)).tolist()
        t = (u, i, j)
        if i != j and t not in exclude and t not in triplets:
            triplets.add(t)
    return list(triplets)

# === PROXIMITY AKA MIN-MAX ===
def choose_items_by_proximity(X, num_triplets, exclude, k=100):
    """Sample i and j with high user-item proximity."""
    n, m = X.shape
    triplets = set()
    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        scores = X[u]
        top_k = torch.topk(scores, k=min(k, m))[1].tolist()
        bottom_k = torch.topk(-scores, k=min(k, m))[1].tolist()
        i = np.random.choice(top_k)
        j = np.random.choice(bottom_k)
        t = (u, i, j)
        if i != j and t not in exclude and t not in triplets:
            triplets.add(t)
    return list(triplets)

# === MARGIN AKA CLOSE-CALL ===
def choose_items_by_margin(X, num_triplets, exclude, max_attempts=5000_000):
    """
    Sample i and j with a margin between their scores, adaptively adjusting the threshold.
    This version uses NumPy for fast operations and batched attempts for efficiency.
    """
    n, m = X.shape
    exclude = exclude or set()
    triplets = set()

    # === Compute adaptive margin using numpy (fast)
    sample = X[:min(10, n)].cpu().numpy()
    margin = np.mean(np.max(sample, axis=1) - np.min(sample, axis=1))*num_triplets/(n*m)

    # === Prepare for efficient sampling
    X_np = X.cpu().numpy()
    rng = np.random.default_rng()

    attempts = 0
    while len(triplets) < num_triplets and attempts < max_attempts:
        batch_size = 500  # Number of triplets to attempt at once
        users = rng.integers(0, n, size=batch_size)
        items = rng.integers(0, m, size=(batch_size, 2))

        for u, (i, j) in zip(users, items):
            if i == j:
                continue
            score_diff = abs(X_np[u, i] - X_np[u, j])
            if score_diff <= margin:
                t = (u, i, j)
                if t not in triplets and t not in exclude:
                    triplets.add(t)
                    if len(triplets) >= num_triplets:
                        break
        attempts += batch_size

    if len(triplets) < num_triplets:
        print(f"⚠️ Only {len(triplets)} triplets generated (target={num_triplets}, margin={margin:.4f}) after {attempts} attempts.maximum : {np.max(X_np)}")

    return list(triplets)

# === VARIANCE (Not used)===
def choose_items_by_variance(X, num_triplets, exclude):
    """Sample i and j with high item variance across users."""
    n, m = X.shape
    variances = torch.var(X, dim=0)
    probs = variances / variances.sum()
    triplets = set()
    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        i, j = torch.multinomial(probs, 2, replacement=False).tolist()
        t = (u, i, j)
        if t not in triplets and t not in exclude:
            triplets.add(t)
    return list(triplets)


# === POPULARITY ===
def choose_items_by_popularity(X, num_triplets, exclude, method="zipf", alpha=1.5):
    """
    Choose i, j based on item popularity distribution.
    Popularity is defined by the method parameter.
    """
    n, m = X.shape
    triplets = set()
    if method == "zipf":
        probs = 1 / (np.arange(1, m + 1) ** alpha)
        probs /= probs.sum()
    elif method == "exponential":
        probs = np.exp(-alpha * np.arange(m))
        probs /= probs.sum()
    elif method == "uniform":
        probs = np.ones(m) / m
    else:
        raise ValueError(f"Unknown popularity method: {method}")

    items = np.arange(m)
    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        i, j = np.random.choice(items, size=2, replace=False, p=probs).tolist()
        t = (u, i, j)
        if i != j and t not in exclude and t not in triplets:
            triplets.add(t)
    return list(triplets)

# === SVD ===
def choose_items_by_svd_projection(X, num_triplets, exclude, rank=10, top_fraction=0.3):
    """
    Use truncated SVD to project users/items and sample (u, i, j)
    from users and items with high singular vector magnitudes (importance in latent space).

    Parameters:
    - X (Tensor): input matrix (n_users x n_items)
    - num_triplets (int): number of triplets to return
    - exclude (set): set of triplets to avoid
    - rank (int): number of top singular values to keep
    - top_fraction (float): fraction of top users/items to keep based on projection norms
    """
    import scipy.sparse.linalg as spla
    rank = int(num_triplets/(X.shape[0]*X.shape[1])* (np.max(X.shape)))
    n, m = X.shape
    X_np = X.cpu().numpy()

    # Truncated SVD
    U, S, Vt = spla.svds(X_np, k=rank)
    U_proj = U @ np.diag(S)       # shape: (n x k)
    V_proj = (np.diag(S) @ Vt).T  # shape: (m x k)

    # Norms (importance) in latent space
    user_norms = np.linalg.norm(U_proj, axis=1)
    item_norms = np.linalg.norm(V_proj, axis=1)

    # Select top users and items
    num_top_users = max(1, int(top_fraction * n))
    num_top_items = max(2, int(top_fraction * m))

    top_users = np.argsort(user_norms)[-num_top_users:]
    top_items = np.argsort(item_norms)[-num_top_items:]

    # Sample triplets
    triplets = set()
    rng = np.random.default_rng()
    for _ in range(num_triplets * 5):
        u = int(rng.choice(top_users))
        i, j = rng.choice(top_items, size=2, replace=False)
        t = (u, i, j)
        if i != j and t not in exclude and t not in triplets:
            triplets.add(t)
        if len(triplets) >= num_triplets:
            break

    if len(triplets) < num_triplets:
        print(f"⚠️ Only {len(triplets)} triplets generated (target={num_triplets})")

    return list(triplets)



# === TOP-K AKA TOP_10% ===
import math

def estimate_k(num_triplets):
    return math.ceil((1 + math.sqrt(1 + 8 * num_triplets)) / 2)

def choose_items_top_k(X, num_triplets, exclude, k=None):
    """
    Choose (u, i, j) where i, j are among top-k items for user u.
    If k is None, it is automatically chosen based on num_triplets.
    """
    n, m = X.shape
    triplets = set()

    # Compute k dynamically if not specified
    if k is None:
        k = min(m, max(5, int(0.1 * m)))
        # Or alternatively:
        # k = min(m, estimate_k(num_triplets))
        # k = 500
        # print("this is the k value", k)

    for _ in range(num_triplets * 3):  # Over-generate
        u = torch.randint(0, n, (1,)).item()
        scores = X[u]
        topk = torch.topk(scores, k=k).indices.tolist()

        i = np.random.choice(topk)
        j = np.random.choice(topk)
        while i == j:
            j = np.random.choice(topk)

        t = (u, i, j)
        if t not in triplets and t not in exclude:
            triplets.add(t)
        if len(triplets) >= num_triplets:
            break

    if len(triplets) < num_triplets:
        print(f"⚠️ Only {len(triplets)} triplets generated (target={num_triplets}, k={k})")

    return list(triplets)

import os

# === CLUSTER (Not used) ===
def choose_items_cluster_based(X, num_triplets, exclude, n_clusters=20):
    """Choose i, j from different clusters of items."""
    
    n, m = X.shape
    triplets = set()
    item_vectors = X.T.numpy()
    clusters = KMeans(n_clusters=n_clusters, n_init='auto').fit_predict(item_vectors)
    cluster_items = {c: np.where(clusters == c)[0] for c in range(n_clusters)}
    cluster_ids = list(cluster_items.keys())

    while len(triplets) < num_triplets:
        u = torch.randint(0, n, (1,)).item()
        c1, c2 = np.random.choice(cluster_ids, 2, replace=False)
        i = np.random.choice(cluster_items[c1])
        j = np.random.choice(cluster_items[c2])
        t = (u, i, j)
        if i != j and t not in exclude and t not in triplets:
            triplets.add(t)
    return list(triplets)


# === USER SIMILARITY (Not used) ===
def choose_items_by_user_similarity(X, num_triplets, exclude=None, max_attempts=10000, verbose=False, fallback_random=False, error_if_incomplete=False):
    """
    Generate (u, i, j) triplets using user similarity and preference divergence.
    Fallback to random sampling if not enough triplets are found.

    Args:
        X (Tensor): Preference matrix (n_users x n_items)
        num_triplets (int): Target number of triplets
        exclude (set): Triplets to exclude (optional)
        max_attempts (int): Max number of tries
        verbose (bool): If True, print progress
        fallback_random (bool): Whether to complete with random triplets
        error_if_incomplete (bool): Raise error instead of returning fewer triplets

    Returns:
        list of (u, i, j) triplets
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n, m = X.shape
    X_np = X.numpy()
    similarities = cosine_similarity(X_np)
    np.fill_diagonal(similarities, -1.0)

    exclude = exclude or set()
    rng = np.random.default_rng()

    avg_triplets_per_user = num_triplets // n + 1
    num_candidates = min(n, int(1.5 * n))
    num_neighbors = min(20, max(3, num_triplets // n))
    top_k = max(3, min(m // 10, 10 + num_triplets // (5 * n)))

    if verbose:
        print(f"→ Adaptive config: top_k={top_k}, neighbors/user={num_neighbors}, target={num_triplets}")

    top_k_items = {
        u: torch.topk(X[u], k=min(top_k, m)).indices.tolist()
        for u in range(n)
    }

    triplets = set()
    attempts = 0

    while len(triplets) < num_triplets and attempts < max_attempts:
        u = rng.integers(0, n)
        similar_users = np.argsort(-similarities[u])[:num_neighbors]

        items_u = set(top_k_items[u])
        for v in similar_users:
            items_v = set(top_k_items[v])
            only_u = list(items_u - items_v)
            only_v = list(items_v - items_u)

            if only_u and only_v:
                i = rng.choice(only_u)
                j = rng.choice(only_v)
            elif len(items_u) >= 2:
                i, j = rng.choice(list(items_u), size=2, replace=False)
            else:
                continue

            t = (u, i, j)
            if i != j and t not in triplets and t not in exclude:
                triplets.add(t)
                break

        attempts += 1
        if verbose and attempts % 1000 == 0:
            print(f"{len(triplets)} triplets generated after {attempts} attempts.")

    # === Fallback or error
    if len(triplets) < num_triplets:
        msg = f"⚠️ Only {len(triplets)} triplets generated (target={num_triplets}) after {attempts} attempts."
        if error_if_incomplete:
            raise RuntimeError(msg)
        elif fallback_random:
            from .helpers import choose_items_random  # or adapt import
            needed = num_triplets - len(triplets)
            extra = choose_items_random(X, needed, exclude=triplets.union(exclude))
            triplets.update(extra)
            if verbose:
                print(f"🔁 Completed with {needed} random triplets.")
        else:
            print(msg)

    if verbose:
        print(f"✅ Returned {len(triplets)} triplets.")
    return list(triplets)


############################################
# These Funtions create U^* and V^* with different structures (to create X^*)
############################################


def generate_embeddings(n, m, d, device="cpu"):
    """
    Efficiently generates a low-rank matrix X = U S V^T where S has rank d,
    and U, V are orthogonal matrices.

    Parameters:
    - n1 (int): Number of rows.
    - n2 (int): Number of columns.
    - d (int): Target rank (d <= min(n1, n2)).

    Returns:
    - X (ndarray): Matrix of shape (n1, n2).
    """
    n1 = min(n, m)
    s = np.zeros(n1)
    s[:d] = 1.0 / np.sqrt(d)

    S = np.zeros((n, m))
    S[:n1, :n1] = np.diag(s)
    U = ortho_group.rvs(dim=n)
    V = ortho_group.rvs(dim=m)
    
    X = U @ S @ V.T 
    X = X * np.sqrt(n*m)/2  # Scale to have unit variance
    return torch.tensor(X, dtype=torch.float32, device=device)


def generate_low_rank_matrix(n, m, d, rank, device="cpu"):
    """
    Generates a low-rank matrix X = U diag(S) V^T with orthonormal U and V.
    - U: (n x d), V: (m x d), both orthonormal
    - S: [1, ..., 1, 0, ..., 0] (rank ones)
    """
    # Generate orthonormal U and V
    U_np = ortho_group.rvs(dim=n)[:, :d]  # shape (n x d)
    V_np = ortho_group.rvs(dim=m)[:, :d]  # shape (m x d)

    # Create singular values S
    S = torch.zeros(d)
    S[:rank] = 1.0

    # Convert to torch
    U = torch.tensor(U_np, dtype=torch.float32, device=device)
    V = torch.tensor(V_np, dtype=torch.float32, device=device)
    
    return U, V, S


def generate_clustered_matrix_from_embeddings(n, m, d, n_clusters=5, device="cpu", scale=1.0, shift_strength=0.5):
    """
    Generates a preference matrix X using base embeddings, then shifts item vectors
    toward their cluster centroid while preserving individual variation.

    Parameters:
    - n (int): Number of users.
    - m (int): Number of items.
    - d (int): Latent dimension.
    - n_clusters (int): Number of item clusters.
    - device (str): Device to use ("cpu" or "cuda").
    - scale (float): Final scaling factor of the matrix.
    - shift_strength (float): Value in [0,1]; how strongly each item is moved toward its cluster mean.

    Returns:
    - X_clustered (Tensor): Matrix of shape (n, m), structured with soft clustering of items.
    """
    # Step 1: Generate initial matrix
    X = generate_embeddings(n, m, d, device=device).clone().detach()  # shape (n, m)
    # Step 2: Cluster item vectors (each column) in X.T
    item_vectors = X.T.cpu().numpy()  # shape (m, n)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(item_vectors)

    # Step 3: Soft shift each item vector toward its cluster center
    X_np = X.cpu().numpy()
    X_shifted = X_np.copy()

    for cluster_id in range(n_clusters):
        item_indices = np.where(labels == cluster_id)[0]
        if len(item_indices) == 0:
            continue
        cluster_mean = X_np[:, item_indices].mean(axis=1, keepdims=True)  # shape (n, 1)
        for j in item_indices:
            original = X_np[:, j:j+1]  # shape (n, 1)
            X_shifted[:, j:j+1] = (1 - shift_strength) * original + shift_strength * cluster_mean

    # Step 4: Convert back to tensor and apply global scaling
    X_clustered = torch.tensor(X_shifted, dtype=torch.float32, device=device) * scale

    return X_clustered


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

    return U/d, V/d


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
    # Base low-d signal
    d_eff=min(d,2)

    U_low = torch.randn(n, d_eff, device=device)
    V_low = torch.randn(m, d_eff, device=device)
    social_influence=0.3

    noise=0.1,
    # Build social graph
    G = nx.watts_strogatz_graph(n, k=5, p=0.1)

    # Apply social influence on U_low
    for u in range(n):
        friends = list(G.neighbors(u))
        if friends:
            friend_mean = torch.stack([U_low[f] for f in friends]).mean(dim=0)
            U_low[u] = (1 - social_influence) * U_low[u] + social_influence * friend_mean

    # Extend with noise in remaining dimensions
    U = torch.cat([U_low, noise * torch.randn(n, d - d_eff, device=device)], dim=1)
    V = torch.cat([V_low, noise * torch.randn(m, d - d_eff, device=device)], dim=1)

    # Normalize V for consistent scale
    V = V / np.sqrt(d)

    # Return final preference matrix
    
    return U, V


def generate_social_embeddings(n, m, d, social_influence=0.5, device="cpu"):
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

    return U/np.log(d+1), V


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

    return U, V/np.sqrt(d)

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
    U = (group_embeddings[user_group_assignments] + 10 * torch.randn(n, d, device=device))

    # Generate structured item embeddings
    V = torch.randn(m, d, device=device)  # Items remain independent

    return U, V/np.log(d+1)


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
# These Funtions compute the preference score
# These functions are never used but still here for some reason...
############################################


def sigmoid_preference(U, V, u, i, j, scale=1.0):
    """
    Computes preference using a sigmoid function, returning 1 or 0.
    """
    return int(torch.sigmoid(scale * torch.dot(U[u], (V[i] - V[j]))).item() > 0.5)

def softmax_preference(U, V, u, i, j, temp=1.0):
    """
    Computes a preference using a softmax over multiple items, returning 1 or 0.
    """
    scores = torch.matmul(V, U[u]) / temp
    probs = torch.softmax(scores, dim=0)
    return int(probs[i].item() > probs[j].item())

def max_preference(U, V, u, i, j):
    """
    Returns 1 if item i is preferred over j, otherwise 0.
    """
    score = torch.dot(U[u], (V[i] - V[j]))
    return int(score.item() > 0)


