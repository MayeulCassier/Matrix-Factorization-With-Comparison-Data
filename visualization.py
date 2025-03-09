# all imports

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd


############################################
# These functions are used to visualize the results, parameters or to get the best parameters
############################################

import matplotlib.pyplot as plt
import numpy as np

def plot_losses(results, param_index=None, selected_indices=None, save_path=""):
    """
    Plots training and validation losses with improved labeling and display of parameter variations.
    
    - Displays only the last repetition of each experiment.
    - Shows only the varying parameters in a separate label box, colored to match the curves.
    - Organizes labels in 4 columns (left to right).
    - Allows filtering experiments using `selected_indices`.
    - Can save the plots as PNG if `save_path` is provided.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_index (int, optional): The index of a specific experiment to plot train & val losses on the same graph.
    - selected_indices (list, optional): List of experiment indices to plot. If None, all experiments are plotted.
    - save_path (str, optional): File name to save the plots. If empty, no saving.
    """
    
    def format_params(params):
        """Format experiment parameters for display."""
        return ", ".join(f"{key}: {value}" for key, value in params.items())

    def find_varying_params(results):
        """Identify parameters that vary across experiments."""
        all_keys = results[0]['params'].keys()
        varying_params = {key for key in all_keys if len(set(exp['params'][key] for exp in results)) > 1}
        return list(varying_params)

    if param_index is not None:
        # Plot a specific experiment's train and validation loss on the same graph
        exp = results[param_index]
        formatted_params = format_params(exp['params'])

        plt.figure(figsize=(10, 5))
        plt.plot(exp['results']['train_losses'][-1], label='Train Loss', linestyle='--')
        plt.plot(exp['results']['val_losses'][-1], label='Val Loss')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train & Val Loss for\n{formatted_params}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
        if save_path:
            plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        
        plt.show()
    
    else:
        # Determine varying parameters
        varying_params = find_varying_params(results)
        param_names = ", ".join(varying_params)

        # Filter indices if selected_indices is provided
        if selected_indices is None:
            selected_indices = range(len(results))
        
        # Prepare colors for matching labels to plots
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

        # Préparer les labels formatés
        param_texts = []
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            param_values = ", ".join(f"{key}={exp['params'][key]}" for key in varying_params)
            param_texts.append((colors[i], f"Exp {exp_idx+1}: {param_values}"))

        # Fonction pour afficher les labels en 4 colonnes
        def display_labels():
            num_cols = 4  # Nombre de colonnes
            x_positions = [0.02, 0.27, 0.52, 0.77]  # Positions en X pour chaque colonne

            for i, (color, text) in enumerate(param_texts):
                row = i // num_cols  # Quelle ligne ?
                col = i % num_cols   # Quelle colonne ?
                y_position = -0.07 - row * 0.05  # Décalage progressif vers le bas
                plt.figtext(x_positions[col], y_position, text, color=color, fontsize=9, ha="left")

        # Plot all selected train losses
        plt.figure(figsize=(10, 5))
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            plt.plot(exp['results']['train_losses'][-1], color=colors[i])

        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title(f"Losses for the parameter scan of the variables:\n {param_names}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        
        display_labels()  # Afficher les labels

        if save_path:
            plt.savefig(f"{save_path}_train.png", bbox_inches="tight", dpi=300)

        plt.show()

        # Plot all selected validation losses
        plt.figure(figsize=(10, 5))
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            plt.plot(exp['results']['val_losses'][-1], color=colors[i])

        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title(f"Losses for the parameter scan of the variables:\n {param_names}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        display_labels()  # Afficher les labels

        if save_path:
            plt.savefig(f"{save_path}_val.png", bbox_inches="tight", dpi=300)

        plt.show()


def plot_heatmap_best_fixed(results, param_x, param_y, result_metric, save_path=""):
    """
    Finds the best configuration for a given result metric and generates a heatmap
    by fixing all parameters except param_x and param_y.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to optimize (maximize for accuracy, minimize for loss/error).
    - save_path (str, optional): If provided, saves the figure as a PNG.
    """
    
    # 🔹 **Déterminer si on doit minimiser (loss/error) ou maximiser**
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()

    # 🔹 **Trouver la meilleure expérience selon la minimisation ou maximisation**
    best_exp_index = min(range(len(results)), key=lambda i: min(results[i]['results'][result_metric])) if is_loss else \
                     max(range(len(results)), key=lambda i: max(results[i]['results'][result_metric]))

    best_exp = results[best_exp_index]
    best_params = best_exp['params']
    best_value = min(best_exp['results'][result_metric]) if is_loss else max(best_exp['results'][result_metric])

    print(f"Best configuration found: {best_params}, {result_metric}: {best_value} (Index: {best_exp_index})")

    # 🔹 **Préparer les données pour la heatmap**
    data = {}
    
    for exp in results:
        # Vérifier que tous les paramètres sont fixés sauf param_x et param_y
        if all(exp['params'][key] == best_params[key] for key in best_params if key not in [param_x, param_y]):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            metric_val = min(exp['results'][result_metric]) if is_loss else max(exp['results'][result_metric])
            
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = metric_val
            else:
                data[(x_val, y_val)] = min(data[(x_val, y_val)], metric_val) if is_loss else max(data[(x_val, y_val)], metric_val)
    
    # 🔹 **Construire la matrice pour la heatmap**
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    
    for (x, y), value in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = value

    # 🔹 **Tracer la heatmap**
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heatmap_matrix, xticklabels=[], yticklabels=[], annot=True, cmap="coolwarm", ax=ax)

    # 🔹 **Forcer la notation scientifique manuellement**
    x_labels_sci = [f"{v:.1e}" for v in x_values]
    y_labels_sci = [f"{v:.1e}" for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels(x_labels_sci, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(y_labels_sci)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y}\n(Best at Index {best_exp_index})")

    # 🔹 **Enregistrer l'image si save_path est fourni**
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    plt.show()




# heatmap between two parameters for maximum value of a result metric with fixed values for other parameters
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index, ax=None, save_path=""):
    """
    Plots a heatmap of two chosen parameters against a selected result metric, with fixed values for other parameters.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    - fixed_index (int): The index of the experiment that should be used to fix other parameters.
    - ax (matplotlib.axes.Axes, optional): Axes on which to plot. If None, creates a new figure.
    - save_path (str, optional): If provided and ax is None, saves the figure as a PNG.
    """
    fixed_params = results[fixed_index]['params']
    data = {}

    # Extract data for heatmap
    for exp in results:
        if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            metric_val = max(exp['results'][result_metric])  # Take the best result for that config
            
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = metric_val
            else:
                data[(x_val, y_val)] = max(data[(x_val, y_val)], metric_val)
    
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    
    for (x, y), value in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = value

    # Flag to track if we're in standalone mode
    standalone_mode = False

    # If no `ax` is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))  # Create a standalone figure
        standalone_mode = True  # We created ax, so we're in standalone mode

    # Plot the heatmap on the specified Axes
    sns.heatmap(heatmap_matrix, xticklabels=[], yticklabels=[], annot=True, cmap="coolwarm", ax=ax)

    # Set labels and title
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y} (Fixed at Index {fixed_index})")

    # 🔹 Convertir manuellement les valeurs en notation scientifique
    x_labels_sci = [f"{v:.1e}" for v in x_values]  
    y_labels_sci = [f"{v:.1e}" for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)  
    ax.set_xticklabels(x_labels_sci, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(y_labels_sci)

    # 🔹 **Enregistrer l'image si save_path est fourni et si c'est en mode standalone**
    if standalone_mode and save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    # Show the plot only if we created the figure
    if standalone_mode:
        plt.show()



# display the indices of the experiments
def display_experiment_indices(results):
    """
    Displays the index and corresponding parameters for each experiment.
    
    Parameters:
    - results (list): The output from parameter_scan.
    """
    print("\nAvailable Experiments:")
    print("Index | Parameters")
    print("--------------------------------------")
    
    for idx, exp in enumerate(results):
        params = exp['params']
        params_str = ', '.join(f"{key}={value}" for key, value in params.items())
        print(f"{idx:<5} | {params_str}")
    print("\nUse these indices to select experiments in other functions like plot_losses or plot_heatmap_fixed.")

# plot the 3D scatter plot for three selected parameters with a result metric as color
def plot_3d_scatter(results, param_x, param_y, param_z, result_metric):
    """
    Plots a 3D scatter plot for three selected parameters with a result metric as color.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter (X-axis).
    - param_y (str): Name of the second parameter (Y-axis).
    - param_z (str): Name of the third parameter (Z-axis).
    - result_metric (str): The result metric to use for coloring the scatter plot.
    """
    data = []
    
    for exp in results:
        data.append({
            param_x: exp['params'][param_x],
            param_y: exp['params'][param_y],
            param_z: exp['params'][param_z],
            result_metric: max(exp['results'][result_metric])  # Take best result
        })
    
    df = pd.DataFrame(data)
    
    # Use Plotly for interactive 3D scatter plot
    fig = px.scatter_3d(df, x=param_x, y=param_y, z=param_z, color=result_metric,
                         title=f"3D Scatter of {result_metric} by {param_x}, {param_y}, and {param_z}",
                         labels={param_x: param_x, param_y: param_y, param_z: param_z, result_metric: result_metric},
                         opacity=0.8)
    fig.show()

# Gets best hyperparameter configuration for a given result metric
def get_best_params(results, result_metric):
    """
    Finds the best hyperparameter configuration for a given result metric.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - result_metric (str): The result metric to optimize.
    
    Returns:
    - dict: The best hyperparameter configuration.
    - int: The index of the best experiment in `results`.
    """
    # determine if we minimize or maximize the result metric
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()
    best_exp_index = min(range(len(results)), key=lambda i: min(results[i]['results'][result_metric])) if is_loss else \
                     max(range(len(results)), key=lambda i: max(results[i]['results'][result_metric]))
    
    best_exp = results[best_exp_index]
    best_params = best_exp['params']
    best_value = min(best_exp['results'][result_metric]) if is_loss else max(best_exp['results'][result_metric])

    print(f"Best parameters for {result_metric} (Index: {best_exp_index}): {best_params}, Best value: {best_value} ")
    return best_params, best_exp_index

# Gets best hyperparameter configuration for all result metrics
def get_best_params_all_metrics(results):
    """
    Finds the best hyperparameter configuration for each result metric in the dataset.
    
    Parameters:
    - results (list): The output from parameter_scan.
    
    Returns:
    - dict: A dictionary where keys are result metrics and values are tuples (best parameter configuration, best experiment index).
    """
    all_metrics = results[0]['results'].keys()  # Get all result metrics from the first experiment
    best_params_per_metric = {}
    
    for metric in all_metrics:
        best_params_per_metric[metric] = get_best_params(results, metric)
    
    return best_params_per_metric
