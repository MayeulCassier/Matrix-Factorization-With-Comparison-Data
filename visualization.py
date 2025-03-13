# all imports

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


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

        # labels formated as (color, text)
        param_texts = []
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            param_values = ", ".join(f"{key}={exp['params'][key]}" for key in varying_params)
            param_texts.append((colors[i], f"Exp {exp_idx+1}: {param_values}"))

        # Fonction pour afficher les labels en 4 colonnes
        def display_labels():
            num_cols = 4  # number of columns for labels
            x_positions = [0.02, 0.27, 0.52, 0.77]  # Positions in X for each column

            for i, (color, text) in enumerate(param_texts):
                row = i // num_cols  
                col = i % num_cols   
                y_position = -0.07 - row * 0.05  # shift down for each row
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
        
        display_labels()  # print labels

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

        display_labels()  # print labels

        if save_path:
            plt.savefig(f"{save_path}_val.png", bbox_inches="tight", dpi=300)

        plt.show()


def plot_heatmap_best_fixed(results, param_x, param_y, result_metric, save_path="", invert_colors=False, log_scale=False):
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
    
    # determine if we minimize or maximize the result metric
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()

    # find the best experiment and its parameters
    best_exp_index = min(range(len(results)), key=lambda i: min(results[i]['results'][result_metric])) if is_loss else \
                     max(range(len(results)), key=lambda i: max(results[i]['results'][result_metric]))

    best_exp = results[best_exp_index]
    best_params = best_exp['params']
    best_value = min(best_exp['results'][result_metric]) if is_loss else max(best_exp['results'][result_metric])

    print(f"Best configuration found: {best_params}, {result_metric}: {best_value} (Index: {best_exp_index})")

    # prepare the data for the heatmap
    data = {}
    
    for exp in results:
        # verify if the parameters are the same as the best parameters
        if all(exp['params'][key] == best_params[key] for key in best_params if key not in [param_x, param_y]):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            metric_val = min(exp['results'][result_metric]) if is_loss else max(exp['results'][result_metric])
            
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = metric_val
            else:
                data[(x_val, y_val)] = min(data[(x_val, y_val)], metric_val) if is_loss else max(data[(x_val, y_val)], metric_val)
    
    # build the heatmap matrix
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    
    for (x, y), value in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = value
    all_values = []
    # Collect min/max values across all heatmaps
    for fixed_index in range(len(results)):
        fixed_params = results[fixed_index]['params']
        data = {}

        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
                x_val = exp['params'][param_x]
                y_val = exp['params'][param_y]
                metric_val = max(exp['results'][result_metric])  # Best result for that config
                if (x_val, y_val) not in data:
                    data[(x_val, y_val)] = metric_val
                else:
                    data[(x_val, y_val)] = max(data[(x_val, y_val)], metric_val)
        
        all_values.extend(data.values())
    vmin, vmax = np.percentile(all_values, [5, 95])  # Prend les valeurs entre le 5ème et 95ème percentile

    # plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = "coolwarm_r" if invert_colors else "coolwarm"
    # Ensure positive values for log scale
    if log_scale:
        vmin = max(vmin, 1e-5)  # Avoid log(0) issues
        vmax = max(vmax, vmin * 10)  # Ensure vmax is larger than vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)  # Apply log scale
    else:
        norm = None  # Use linear scale

    sns.heatmap(heatmap_matrix, vmin=vmin, vmax=vmax, xticklabels=[], yticklabels=[], annot=True, norm = norm, cmap=cmap, ax=ax, fmt=".4f", annot_kws={"size": 12})

    def format_sci(v):
        """Formate les valeurs en notation scientifique sauf pour les unités"""
        return f"{v:.1e}".replace("e+00", "") if "e" in f"{v:.1e}" else f"{v:.1f}"
    
    # Force scientific notation manually
    x_labels_sci = [format_sci(v) for v in x_values]
    y_labels_sci = [format_sci(v) for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels(x_labels_sci, rotation=45, ha="right", fontsize=12)

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(y_labels_sci, fontsize=12)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y}\n(Best at Index {best_exp_index})"
                 f"{'Log Scale' if log_scale else 'Linear Scale'}")
    ax.tick_params(axis='both', labelsize=12)

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    plt.show()

# plot the heatmap for two selected parameters with a result metric
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index, ax=None, save_path="", log_scale=False, invert_colors=False):
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
    - log_scale (bool, optional): If True, applies a logarithmic scale to the colormap.
    - invert_colors (bool, optional): If True, reverses the colormap.
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

    # Define color normalization (logarithmic only for colors, not values)
    vmin, vmax = np.percentile(list(data.values()), [5, 95])  # Avoid extreme outliers
    if log_scale:
        vmin = max(vmin, 1e-5)  # Prevent log(0) errors
        vmax = max(vmax, vmin * 10)  # Ensure vmax is larger
        norm = LogNorm(vmin=vmin, vmax=vmax)  # Apply log scale
    else:
        norm = None  # Linear scale

    # Choose colormap (invert if requested)
    cmap = "coolwarm_r" if invert_colors else "coolwarm"

    # Plot the heatmap with optional log color scale
    sns.heatmap(heatmap_matrix, cmap=cmap, norm=norm, ax=ax, annot=True, fmt=".4f",
                annot_kws={"size": 12})

    # Set labels and title
    ax.set_xlabel(param_x, fontsize=14)
    ax.set_ylabel(param_y, fontsize=14)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y} (Fixed at Index {fixed_index})\n"
                 f"{'Log Colors' if log_scale else 'Linear Colors'} - {'Inverted' if invert_colors else 'Normal'}")

    def format_sci(v):
        """Format values in scientific notation if needed"""
        return f"{v:.1e}".replace("e+00", "") if "e" in f"{v:.1e}" else f"{v:.1f}"
    
    # Format labels in scientific notation
    x_labels_sci = [format_sci(v) for v in x_values]
    y_labels_sci = [format_sci(v) for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)  
    ax.set_xticklabels(x_labels_sci, rotation=45, ha="right", fontsize=12)

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(y_labels_sci, fontsize=12)

    ax.tick_params(axis='both', labelsize=12)

    # Save the plot if we're in standalone mode
    if standalone_mode and save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    # Show the plot only if we created the figure
    if standalone_mode:
        plt.show()



def find_fixed_indices(results, param_x, param_y):
    """
    Finds the first indices where fixed parameters (excluding param_x and param_y) change.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first varying parameter.
    - param_y (str): Name of the second varying parameter.

    Returns:
    - list: Indices of the first occurrences of each unique fixed parameter combination.
    """
    unique_fixed_params = {}
    fixed_indices = []

    for idx, exp in enumerate(results):
        # filter only the fixed parameters (excluding param_x and param_y)
        fixed_params = tuple((k, v) for k, v in exp['params'].items() if k not in [param_x, param_y])

        # if the fixed parameters are not in the dictionary, add them
        if fixed_params not in unique_fixed_params:
            unique_fixed_params[fixed_params] = idx
            fixed_indices.append(idx)

    return fixed_indices

# plot all heatmaps
def plot_all_heatmaps(results, param_x, param_y, result_metric, fig_size=(12, 10), save_path="", invert_colors=False, log_scale=False):   
    """
    Plots all heatmaps in a grid with the same color scale.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    - fig_size (tuple, optional): Size of the figure.
    - save_path (str, optional): If provided, saves the figure as a PNG.
    
    """

    all_indices = find_fixed_indices(results, param_x, param_y)
    plot_multiple_heatmaps(results, param_x, param_y, result_metric, all_indices, fig_size, save_path, invert_colors=invert_colors, log_scale=log_scale)





# plot multiple heatmaps in a grid with the same color scale
def plot_multiple_heatmaps(results, param_x, param_y, result_metric, fixed_indices = None, fig_size=(12, 10), save_path="", invert_colors=False, log_scale=False):
    """
    Plots multiple heatmaps in a grid with the same color scale.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    - fixed_indices (list): List of experiment indices that should be used to fix other parameters.
    - fig_size (tuple, optional): Size of the figure.
    - save_path (str, optional): If provided, saves the figure as a PNG.
    """
    
    
    if fixed_indices is None:
        fixed_indices = find_fixed_indices(results, param_x, param_y)

    if len(fixed_indices) == 1:
        plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_indices[0], save_path=save_path, log_scale=log_scale, invert_colors=invert_colors)
        return
    num_rows = len(fixed_indices) // 2 + (len(fixed_indices) % 2)  # Arrange subplots in a 2-column layout
    num_cols = 2  # Maximum 2 heatmaps per row

    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size, constrained_layout=True)
    axes = axes.flatten()  # Flatten in case of single row/column layout

    all_values = []  # Collect values for common color scale
    fixed_params_list = [results[idx]['params'] for idx in fixed_indices]  # Extract fixed params for each plot

    # find the varying and constant keys
    all_keys = set(fixed_params_list[0].keys())  # All keys in the first fixed params
    constant_keys = {key for key in all_keys if all(d[key] == fixed_params_list[0][key] for d in fixed_params_list)}
    varying_keys = all_keys - constant_keys  # Varying keys are the rest

    # Collect min/max values across all heatmaps
    for fixed_index in fixed_indices:
        fixed_params = results[fixed_index]['params']
        data = {}

        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
                x_val = exp['params'][param_x]
                y_val = exp['params'][param_y]
                metric_val = max(exp['results'][result_metric])  # Best result for that config
                if (x_val, y_val) not in data:
                    data[(x_val, y_val)] = metric_val
                else:
                    data[(x_val, y_val)] = max(data[(x_val, y_val)], metric_val)
        
        all_values.extend(data.values())

    vmin, vmax = np.percentile(all_values, [5, 95])   # Define common scale
    # Ensure positive values for log scale
    if log_scale:
        vmin = max(vmin, 1e-5)  # Avoid log(0) issues
        vmax = max(vmax, vmin * 10)  # Ensure vmax is larger than vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)  # Apply log scale
    else:
        norm = None  # Use linear scale

    # Plot each heatmap with the same color scale
    for i, fixed_index in enumerate(fixed_indices):
        fixed_params = results[fixed_index]['params']
        data = {}

        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
                x_val = exp['params'][param_x]
                y_val = exp['params'][param_y]
                metric_val = max(exp['results'][result_metric])
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

        def format_sci(v):
            """Formate les valeurs en notation scientifique sauf pour les unités"""
            return f"{v:.1e}".replace("e+00", "") if "e" in f"{v:.1e}" else f"{v:.1f}"
    
        # Force scientific notation manually
        x_labels_sci = [format_sci(v) for v in x_values]
        y_labels_sci = [format_sci(v) for v in y_values]

        cmap = "coolwarm_r" if invert_colors else "coolwarm"

        # Plot heatmap
        sns.heatmap(
            heatmap_matrix, xticklabels=x_labels_sci, yticklabels=y_labels_sci, annot=True, cmap=cmap,
            ax=axes[i], vmin=vmin, vmax=vmax,norm =norm, fmt=".4f", annot_kws={"size": 10}, cbar=i % 2 == 1
        )

        # Set titles and labels
        varying_params_str = ", ".join(f"{key}={fixed_params[key]}" for key in varying_keys)

        # Set titles and labels
        axes[i].set_xlabel(param_x, fontsize=12)
        axes[i].set_ylabel(param_y, fontsize=12)
        axes[i].set_title(f"Fixed: Index {fixed_index}\n{varying_params_str}"
                          f"{'Log Scale' if log_scale else 'Linear Scale'}", fontsize=14)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

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

# print the results for indices and if not specified, prints all the results 

def print_results(results, indices=None, params_off=False, metric=None):
    """
    Prints the results of the experiments with optional filtering and display options.

    Parameters:
    - results (list): The output from parameter_scan.
    - indices (list, optional): List of experiment indices to display. If None, displays all experiments.
    - params_off (bool, optional): If True, does not display the parameters for each experiment.
    - metric (str, optional): The result metric to display. If None, displays all metrics.
    """

    if indices is None:
        indices = range(len(results))
    if metric is None:
        metric = list(results[0]['results'].keys())[0]
    # Display results for each experiment
    for idx in indices:
        exp = results[idx]
        params_str = "" if params_off else f"Params: {exp['params']}"
        print(f"Index: {idx}, {params_str}, {metric}: {exp['results'][metric]}")
