import torch # type: ignore
import networkx as nx
from sklearn.mixture import GaussianMixture


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
# These Funtions choose i, j for preference score
############################################

import numpy as np

def choose_items_random(m):
    """
    Randomly selects two different items.
    """
    i, j = torch.randint(0, m, (2,)).tolist()
    while i == j:
        j = torch.randint(0, m, (1,)).item()
    return i, j

def choose_items_by_proximity(U, V, u):
    """
    Selects two items that are closest to the user's embedding.
    """
    scores = torch.matmul(V, U[u])  # Compute similarity scores
    sorted_indices = torch.argsort(scores, descending=True)
    return sorted_indices[0].item(), sorted_indices[-1].item()  # Most and least similar

def choose_items_by_variance(V):
    """
    Selects one item with high variance in embeddings and one with low variance.
    """
    variances = torch.var(V, dim=1)
    i = torch.argmax(variances).item()
    j = torch.argmin(variances).item()
    return i, j

def choose_items_by_popularity(m, method="zipf", alpha=1.5):
    """
    Chooses items based on a popularity distribution.
    """
    popularity_dist = generate_popularity_distribution(m, method, alpha)
    i=j=0
    while i==j:
        i = np.random.choice(len(popularity_dist), p=popularity_dist)
        j = np.random.choice(len(popularity_dist), p=1 - popularity_dist)
    return i, j 

import numpy as np

def generate_popularity_distribution(m, method="zipf", alpha=1.5):
    """
    Generates a popularity distribution for items.
    
    Parameters:
    - m (int): Number of items.
    - method (str): "uniform", "zipf", or "exponential".
    - alpha (float): Parameter controlling skewness (only for Zipf and exponential).
    
    Returns:
    - np.array: Probability distribution over items.
    """
    if method == "uniform":
        return np.ones(m) / m  # Equal probability for all items
    elif method == "zipf":
        ranks = np.arange(1, m + 1)
        probs = 1 / ranks ** alpha
        return probs / probs.sum()  # Normalize to sum to 1
    elif method == "exponential":
        probs = np.exp(-alpha * np.arange(m))
        return probs / probs.sum()  # Normalize to sum to 1
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'zipf', or 'exponential'.")


############################################
# These Funtions compute the preference score
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


