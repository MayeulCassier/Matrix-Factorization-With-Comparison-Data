# all imports

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import sem


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


def plot_heatmap_best_fixed(results, param_x, param_y, result_metric, save_path="", invert_colors=False, log_scale=False, ignored_keys=None):
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
    ignored_keys = ignored_keys or []
    
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
        if all(exp['params'][key] == best_params[key] for key in best_params if key not in [param_x, param_y]+ignored_keys):
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
    all_min_values = []
    all_max_values = []

    for fixed_index in range(len(results)):
        fixed_params = results[fixed_index]['params']
        
        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]+ignored_keys):
                values = exp['results'][result_metric]
                all_min_values.append(min(values))
                all_max_values.append(max(values))

    vmin = np.percentile(all_min_values, 5)
    vmax = np.percentile(all_max_values, 95)
    
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
        """Format the value in scientific notation only for very small or very large numbers"""
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
        else:
            return f"{v:.2f}".rstrip('0').rstrip('.')  # Remove trailing zeros and dots if needed

    # Force scientific notation manually
    x_labels_sci = [format_sci(v) for v in x_values]
    y_labels_sci = [format_sci(v) for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels(x_labels_sci, rotation=45, ha="right", fontsize=12)

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(y_labels_sci, fontsize=12)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y}")
    ax.tick_params(axis='both', labelsize=12)

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    plt.show()

# plot the heatmap for two selected parameters with a result metric
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index, ax=None, save_path="", log_scale=False, invert_colors=False, ignored_keys=None):
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
    ignored_keys = ignored_keys or []

    # Extract data for heatmap
    for exp in results:
        if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y] + ignored_keys):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            values = exp['results'][result_metric]
            mean_val = np.mean(values)
            err_val = sem(values) if len(values) > 1 else 0  # Standard error of the mean
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = (mean_val, err_val)
            else:
                prev_mean, prev_err = data[(x_val, y_val)]
                data[(x_val, y_val)] = ((prev_mean + mean_val)/2, (prev_err + err_val)/2)    
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    
    annot_matrix = np.empty_like(heatmap_matrix, dtype=object)
    for (x, y), (mean_val, err_val) in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = mean_val
        if err_val > 0:
            annot_matrix[y_idx, x_idx] = f"{mean_val:.4f}\n±{err_val:.4f}"
        else:
            annot_matrix[y_idx, x_idx] = f"{mean_val:.4f}"

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
    sns.heatmap(heatmap_matrix, cmap=cmap, norm=norm, ax=ax,
                annot=annot_matrix, fmt='', annot_kws={"size": 12})

    # Set labels and title
    ax.set_xlabel(param_x, fontsize=14)
    ax.set_ylabel(param_y, fontsize=14)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y}")

    def format_sci(v):
        """Format the value in scientific notation only for very small or very large numbers"""
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
        else:
            return f"{v:.2f}".rstrip('0').rstrip('.')  # Remove trailing zeros and dots if needed

    
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



def find_fixed_indices(results, param_x, param_y, ignored_keys=None):
    """
    Finds the first indices where fixed parameters (excluding param_x and param_y) change.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first varying parameter.
    - param_y (str): Name of the second varying parameter.
    - ignored_keys (list, optional): List of keys to ignore when determining fixed parameters.

    Returns:
    - list: Indices of the first occurrences of each unique fixed parameter combination.
    """
    ignored_keys = ignored_keys or []
    unique_fixed_params = {}
    fixed_indices = []

    for idx, exp in enumerate(results):
        fixed_params = tuple(
            (k, v) for k, v in exp['params'].items()
            if k not in [param_x, param_y] + ignored_keys
        )
        if fixed_params not in unique_fixed_params:
            unique_fixed_params[fixed_params] = idx
            fixed_indices.append(idx)

    return fixed_indices

# plot all heatmaps
def plot_all_heatmaps(results, param_x, param_y, result_metric,
                      fig_size=(12, 10), save_path="", invert_colors=False,
                      log_scale=False, ignored_keys=None):
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
    
    all_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)
    plot_multiple_heatmaps(results, param_x, param_y, result_metric, all_indices,
                           fig_size, save_path, invert_colors, log_scale, ignored_keys=ignored_keys)
   


def enrich_params_with_data_points(results):
    """
    Adds 'num_data_points' = n * m * p * 0.5 to each experiment's params.
    """
    for exp in results:
        n = exp['params']['n']
        m = exp['params']['m']
        p = exp['params']['p']
        exp['params']['num_data_points'] = n * m * p * 0.5
        # Round num_data_points to avoid float comparison issues
        exp['params']['num_data_points'] = round(exp['params']['num_data_points'], 4)

    return results



# plot multiple heatmaps in a grid with the same color scale
def plot_multiple_heatmaps(results, param_x, param_y, result_metric,
                           fixed_indices=None, fig_size=(12, 10), save_path="",
                           invert_colors=False, log_scale=False, ignored_keys=None):
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
    
    
    ignored_keys = ignored_keys or []

    if fixed_indices is None:
        fixed_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)

    if len(fixed_indices) == 1:
        plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_indices[0],
                           save_path=save_path, log_scale=log_scale, invert_colors=invert_colors, ignored_keys=ignored_keys)
        return
    num_rows = len(fixed_indices) // 2 + (len(fixed_indices) % 2)  # Arrange subplots in a 2-column layout
    num_cols = 2  # Maximum 2 heatmaps per row

    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size, constrained_layout=True)
    axes = axes.flatten()  # Flatten in case of single row/column layout

    fixed_params_list = [results[idx]['params'] for idx in fixed_indices]  # Extract fixed params for each plot
    # find the varying and constant keys
    all_keys = set(fixed_params_list[0].keys())  # All keys in the first fixed params
    constant_keys = {key for key in all_keys if all(d[key] == fixed_params_list[0][key] for d in fixed_params_list)}
    varying_keys = all_keys - constant_keys  # Varying keys are the rest

    all_min_values = []
    all_max_values = []
    for fixed_index in fixed_indices:
        fixed_params = results[fixed_index]['params']
        
        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
                values = exp['results'][result_metric]
                all_min_values.append(min(values))
                all_max_values.append(max(values))

    vmin = np.percentile(all_min_values, 5)
    vmax = np.percentile(all_max_values, 95)

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
            if all(exp['params'].get(key) == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]+ignored_keys):
                x_val = exp['params'][param_x]
                y_val = exp['params'][param_y]
                values = exp['results'][result_metric]
                mean_val = np.mean(values)
                err_val = sem(values) if len(values) > 1 else 0.0
                data[(x_val, y_val)] = (mean_val, err_val)
        x_values = sorted(set(k[0] for k in data.keys()))
        y_values = sorted(set(k[1] for k in data.keys()))
        heatmap_matrix = np.zeros((len(y_values), len(x_values)))
        annot_matrix = np.empty_like(heatmap_matrix, dtype=object)

        for (x, y), (mean_val, err_val) in data.items():
            x_idx = x_values.index(x)
            y_idx = y_values.index(y)
            heatmap_matrix[y_idx, x_idx] = mean_val
            if err_val > 0:
                annot_matrix[y_idx, x_idx] = f"{mean_val:.3f}\n±{err_val:.3f}"
            else:
                annot_matrix[y_idx, x_idx] = f"{mean_val:.3f}"

        def format_sci(v):
            """Format the value in scientific notation only for very small or very large numbers"""
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
                return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
            else:
                return f"{v:.2f}".rstrip('0').rstrip('.')  # Remove trailing zeros and dots if needed

        # Force scientific notation manually
        x_labels_sci = [format_sci(v) for v in x_values]
        y_labels_sci = [format_sci(v) for v in y_values]

        cmap = "coolwarm_r" if invert_colors else "coolwarm"

        # Plot heatmap
        sns.heatmap(
            heatmap_matrix, xticklabels=x_labels_sci, yticklabels=y_labels_sci, annot=annot_matrix, cmap=cmap,
            ax=axes[i], vmin=vmin, vmax=vmax, norm=norm, fmt="", annot_kws={"size": 10}, cbar=i % 2 == 1
        )


        # Set titles and labels
        varying_params_str = ", ".join(f"{key}={fixed_params[key]}" for key in varying_keys)

        # Set titles and labels
        axes[i].set_xlabel(param_x, fontsize=12)
        axes[i].set_ylabel(param_y, fontsize=12)
        axes[i].set_title(f"Heatmap with parameters: \n{varying_params_str}", fontsize=14)

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
        print(f": Index{idx}, {params_str}, {metric}: {exp['results'][metric]}")


import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np
import math
from itertools import product
def plot_metrics_vs_param(results, param_x, metrics, group_by=None,
                          split_by=None, title="", grid=True, save_path=None,
                          ylim=None, log_scale_x=False, log_scale_y=False,
                          sub_plot=True):
    """
    Plots metrics vs a parameter, grouped and split by other hyperparameters.
    
    Parameters:
    - param_x (str): X-axis param.
    - metrics (list): List of metrics to plot.
    - group_by (str/list): Param(s) used to distinguish lines in a plot.
    - split_by (str/list): Param(s) used to create separate plots or subplots.
    - title (str): Plot title.
    - grid (bool): Show grid.
    - save_path (str): Path prefix for saving plots.
    - ylim (tuple): (ymin, ymax).
    - log_scale_x (bool): Log scale on x-axis.
    - log_scale_y (bool): Log scale on y-axis.
    - sub_plot (bool): If True (default), show subplots. Else, show separate figures.
    """
    
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(split_by, str):
        split_by = [split_by]
    group_by = group_by or []
    split_by = split_by or []

    # Define styles per metric
    markers = ['o', 's', 'D', '^', 'v', 'x']
    linestyles = ['-', '--', '-.', ':']
    metric_styles = {
        metric: {
            'marker': markers[i % len(markers)],
            'linestyle': linestyles[i % len(linestyles)]
        } for i, metric in enumerate(metrics)
    }

    

    # === Build all combinations of split_by keys from the data ===
    unique_values = {key: sorted(set(exp['params'].get(key) for exp in results)) for key in split_by}
    all_combinations = list(product(*(unique_values[k] for k in split_by)))

    # === Group results by full match of split_by values ===
    split_groups = {}
    for combo in all_combinations:
        combo_dict = dict(zip(split_by, combo))
        matching_exps = [
            exp for exp in results
            if all(exp['params'].get(k) == v for k, v in combo_dict.items())
        ]
        split_key = tuple((k, combo_dict[k]) for k in split_by)
        if matching_exps:
            split_groups[split_key] = matching_exps


    if sub_plot:
        # === SUBPLOTS MODE ===
        num_plots = len(split_groups)
        ncols = min(2, num_plots)
        nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, (split_key, group_results) in enumerate(split_groups.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                            metric_styles, color_cycle, split_key, split_by,
                            title, grid, ylim, log_scale_x, log_scale_y)

        # Hide unused subplots
        for j in range(num_plots, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        plt.tight_layout()
        if save_path:
            full_path = f"{save_path}.png"
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            print(f"Saved combined subplot figure to: {full_path}")
        plt.show()

    else:
        # === SEPARATE FIGURES MODE ===
        for split_key, group_results in split_groups.items():
            fig, ax = plt.subplots(figsize=(9, 6))
            _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                            metric_styles, plt.cm.tab10(np.linspace(0, 1, 10)),
                            split_key, split_by, title, grid, ylim, log_scale_x, log_scale_y)

            plt.tight_layout()
            if save_path:
                suffix = "_".join(f"{k}_{v}" for k, v in split_key)
                full_path = f"{save_path}_{suffix}.png"
                plt.savefig(full_path, bbox_inches="tight", dpi=300)
                print(f"Saved individual plot to: {full_path}")
            plt.show()


def _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                    metric_styles, color_cycle, split_key, split_by,
                    title, grid, ylim, log_scale_x, log_scale_y):
    """
    Internal helper: plots one panel (subplot or full figure).
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for exp in group_results:
        group_key = tuple((k, exp['params'].get(k, None)) for k in group_by)
        grouped[group_key].append(exp)

    color_map = {group: color_cycle[i % len(color_cycle)] for i, group in enumerate(grouped)}

    for group_key, exps in grouped.items():
        exps_sorted = sorted(exps, key=lambda e: e['params'][param_x])
        x_vals = []
        metric_vals = {metric: [] for metric in metrics}
        metric_errs = {metric: [] for metric in metrics}

        for exp in exps_sorted:
            x = exp['params'][param_x]
            x_vals.append(x)

            for metric in metrics:
                values = exp['results'][metric]
                if isinstance(values[0], list):
                    values = [v[-1] for v in values]
                mean_val = np.mean(values)
                err_val = sem(values) if len(values) > 1 else 0.0
                metric_vals[metric].append(mean_val)
                metric_errs[metric].append(err_val)

        for metric in metrics:
            style = metric_styles[metric]
            label = f"{metric} ({', '.join(f'{k}={v}' for k, v in group_key)})" if group_by else metric
            ax.errorbar(x_vals, metric_vals[metric], yerr=metric_errs[metric],
                        fmt=style['marker'] + style['linestyle'], capsize=5,
                        label=label, color=color_map[group_key])

    split_label = ", ".join(f"{k}={v}" for k, v in split_key) if split_by else ""
    full_title = f"{title}\n{split_label}" if split_label else title
    ax.set_title(full_title)
    ax.set_xlabel(param_x)
    ax.set_ylabel("Metric Value")
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    if ylim:
        ax.set_ylim(ylim)
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")
    ax.legend()
