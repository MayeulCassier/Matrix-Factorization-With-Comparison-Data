# all imports

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import sem
import numpy as np
import math
import matplotlib.ticker as mticker
from itertools import product
from matplotlib.cm import get_cmap
from collections import defaultdict
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})


############################################
# These functions are used to visualize the results, parameters or to get the best parameters
############################################


def format_display_name(name):
    """
    Formats a raw metric or parameter name into a human-readable display string.

    Examples:
    - "gt_log_likelihoods" → "Ground Truth Log Likelihood"
    - "val_losses" → "Validation Loss"
    - "weight_decay" → "Weight Decay"
    - "num_data_points" → "Num Data Points"
    """
    name_map = {
        # Metrics
        "train_losses": "Training Loss",
        "val_losses": "Validation Loss",
        "accuracy": "Accuracy",
        "log_likelihoods": "Log Likelihood",
        "gt_accuracy": "Ground Truth Accuracy",
        "gt_log_likelihoods": "Ground Truth Log Likelihood",
        "reconstruction_errors": "Reconstruction Error",
        "gt_loss": "Ground Truth Loss",

        # Parameters
        "lr": "Learning Rate",
        "weight_decay": "Weight Decay",
        "num_epochs": "Num Epochs",
        "num_data_points": "Num Data Points",
        "p": "p",
        "d": "Embedding Dim (d)",
        "d1": "Init Dim (d1)",
        "K" : "k",
        "n": "n",
        "s": "s",
        "m": "m",
        "norm_ratio": "$\|X^*\|/\|UV^T\|$",
        "reconstruction_error_scaled": "Reconstruction Error (Scaled)",
        "svd_error_scaled": "SVD Error (Scaled)",
        "spearman_corr": "Spearman Correlation",
        "pearson_corr": "Pearson Correlation",

    }

    if name in name_map:
        return name_map[name]
    else:
        return name.replace("_", " ").title()


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
        return ", ".join(f"{format_display_name(key)}: {value}" for key, value in params.items())

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
        plt.plot(exp['results']['train_losses'][-1], label=r'Training Loss', linestyle='--')
        plt.plot(exp['results']['val_losses'][-1], label=r'Validation Loss')

        plt.xlabel(r"Epochs")
        plt.ylabel(r"Loss")
        plt.title(rf"Train \& Val Loss for\\{formatted_params}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
        if save_path:
            plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        
        plt.show()
    
    else:
        # Determine varying parameters
        varying_params = find_varying_params(results)
        param_names = ", ".join(format_display_name(p) for p in varying_params)

        # Filter indices if selected_indices is provided
        if selected_indices is None:
            selected_indices = range(len(results))
        
        # Prepare colors for matching labels to plots
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

        # labels formated as (color, text)
        param_texts = []
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            param_values = ", ".join(f"{format_display_name(key)}={exp['params'][key]}" for key in varying_params)
            param_texts.append((colors[i], f"Exp {exp_idx+1}: {param_values}"))

        # Fonction pour afficher les labels en 4 colonnes
        def display_labels():
            num_cols = 4  # number of columns for labels
            x_positions = [0.02, 0.27, 0.52, 0.77]  # Positions in X for each column

            for i, (color, text) in enumerate(param_texts):
                row = i // num_cols  
                col = i % num_cols   
                y_position = -0.07 - row * 0.05  # shift down for each row
                plt.figtext(x_positions[col], y_position, rf"$\text{{{text}}}$", color=color, fontsize=9, ha="left")

        # Plot all selected train losses
        plt.figure(figsize=(10, 5))
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            plt.plot(exp['results']['train_losses'][-1], color=colors[i])

        plt.xlabel(r"Epochs")
        plt.ylabel(r"Train Loss")
        plt.title(rf"Losses for the parameter scan of the variables:\\ {param_names}", fontsize=12)
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

        plt.xlabel(r"Epochs")
        plt.ylabel(r"Validation Loss")
        plt.title(rf"Losses for the parameter scan of the variables:\\ {param_names}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        display_labels()  # print labels

        if save_path:
            plt.savefig(f"{save_path}_val.png", bbox_inches="tight", dpi=300)

        plt.show()


def plot_heatmap_best_fixed(results, param_x, param_y, result_metric, save_path="", invert_colors=False, log_scale=False, ignored_keys=None, overall=True, invert_x=False, invert_y=False, fig_size=(10, 7), font_scale=1):
    """
    Plots a heatmap of result_metric across param_x and param_y. 

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    - save_path (str, optional): If provided, saves the figure as a PNG.
    - invert_colors (bool, optional): If True, reverses the colormap.
    - log_scale (bool, optional): If True, applies a logarithmic scale to the colormap.
    - ignored_keys (list, optional): List of keys to ignore when determining fixed parameters.
    - If overall=True: restricts to configs matching the best global setting
    - If overall=False: uses all experiments and picks best (mean) value for each (x, y)
    Adds ±SEM annotation if multiple reps per config.


    """
    ignored_keys = ignored_keys or []
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()
    data = {}

    if not overall:
        # === Step 1: Identify best global configuration
        best_exp_index = min(range(len(results)), key=lambda i: min(results[i]['results'][result_metric])) if is_loss else \
                         max(range(len(results)), key=lambda i: max(results[i]['results'][result_metric]))
        best_params = results[best_exp_index]['params']

        # === Step 2: Filter and collect (x, y) values matching best fixed params
        for exp in results:
            if all(exp['params'][k] == best_params[k] for k in best_params if k not in [param_x, param_y] + ignored_keys):
                x = exp['params'][param_x]
                y = exp['params'][param_y]
                values = exp['results'][result_metric]
                mean_val = np.mean(values)
                err_val = sem(values) if len(values) > 1 else 0.0
                key = (x, y)
                if key not in data or (is_loss and mean_val < data[key][0]) or (not is_loss and mean_val > data[key][0]):
                    data[key] = (mean_val, err_val)

    else:
        # === Global max/min across ALL configs for each (x, y), regardless of other params
        for exp in results:
            if param_x not in exp['params'] or param_y not in exp['params']:
                continue
            x = exp['params'][param_x]
            y = exp['params'][param_y]
            values = exp['results'][result_metric]
            mean_val = np.mean(values)
            err_val = sem(values) if len(values) > 1 else 0.0
            key = (x, y)
            if key not in data or (is_loss and mean_val < data[key][0]) or (not is_loss and mean_val > data[key][0]):
                data[key] = (mean_val, err_val)

    # === Step 3: Build heatmap matrix
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    if invert_x:
        x_values = list(reversed(x_values))
    if invert_y:
        y_values = list(reversed(y_values))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    annot_matrix = np.empty_like(heatmap_matrix, dtype=object)

    for (x, y), (mean_val, err_val) in data.items():
        xi = x_values.index(x)
        yi = y_values.index(y)
        heatmap_matrix[yi, xi] = mean_val
        annot_matrix[yi, xi] = f"{mean_val:.4f}\n±{err_val:.4f}" if err_val > 0 else f"{mean_val:.4f}"

    # === Step 4: Plot
    vmin = np.percentile([v[0] for v in data.values()], 5)
    vmax = np.percentile([v[0] for v in data.values()], 95)
    if log_scale:
        vmin = max(vmin, 1e-5)
        vmax = max(vmax, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=fig_size)
    cmap = "coolwarm_r" if invert_colors else "coolwarm"
    heatmap = sns.heatmap(heatmap_matrix, annot=annot_matrix, fmt="", cmap=cmap,
                xticklabels=[], yticklabels=[], norm=norm,
                annot_kws={"size": 12*font_scale}, ax=ax, cbar=True)
    if heatmap.collections and heatmap.collections[0].colorbar:
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=font_scale*12)  # Adjust colorbar font size
    def format_sci(v):
        return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-") if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0) else f"{v:.2f}".rstrip('0').rstrip('.')

    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels([rf"{format_sci(v)}" for v in x_values], rotation=45, ha="right", fontsize=12*font_scale)
    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels([rf"{format_sci(v)}" for v in y_values], fontsize=12*font_scale)

    ax.set_xlabel(rf"{format_display_name(param_x)}", font_size=14*font_scale)
    ax.set_ylabel(rf"{format_display_name(param_y)}", font_size=14*font_scale)
    ax.set_title(rf"Heatmap of {format_display_name(result_metric)} by {format_display_name(param_x)} and {format_display_name(param_y)} ({'global best block' if overall else 'best per (x,y)'})", fontsize=16*font_scale)
    ax.tick_params(axis='both', labelsize=12*font_scale)

    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    plt.show()

def enrich_params_with_data_points(results):
    """
    Adds 'num_data_points' = n * m * p * 0.5 to each experiment's params.
    """
    for exp in results:
        n = exp['params']['n']
        m = exp['params']['m']
        p = exp['params']['p']
        K = exp['params']['K']
        exp['params']['num_data_points'] = n * m * p * 0.5 *K
        # Round num_data_points to avoid float comparison issues
        exp['params']['num_data_points'] = round(exp['params']['num_data_points'], 4)

    return results

# plot the heatmap for two selected parameters with a result metric
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index,
                        save_path="", invert_colors=False, log_scale=False,
                        ignored_keys=None, overall=True,
                        invert_x=False, invert_y=False, ax=None, font_scale=1):
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
    # Invert axis values if requested
    if invert_x:
        x_values = list(reversed(x_values))
    if invert_y:
        y_values = list(reversed(y_values))

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
    mean_vals = [mean for (mean, _) in data.values()]
    vmin, vmax = np.percentile(mean_vals, [5, 95])
    if log_scale:
        vmin = max(vmin, 1e-5)  # Prevent log(0) errors
        vmax = max(vmax, vmin * 10)  # Ensure vmax is larger
        norm = LogNorm(vmin=vmin, vmax=vmax)  # Apply log scale
    else:
        norm = None  # Linear scale

    # Choose colormap (invert if requested)
    cmap = "coolwarm_r" if invert_colors else "coolwarm"

    # Plot the heatmap with optional log color scale
    heatmap = sns.heatmap(heatmap_matrix, cmap=cmap, norm=norm, ax=ax,
                annot=annot_matrix, fmt='', annot_kws={"size": 12*font_scale}, cbar=True)

    # Set labels and title
    ax.set_xlabel(format_display_name(param_x), fontsize=14*font_scale)
    ax.set_ylabel(format_display_name(param_y), fontsize=14*font_scale)
    ax.set_title(f"Heatmap of {format_display_name(result_metric)} by {format_display_name(param_x)} and {format_display_name(param_y)}", fontsize=16*font_scale)

    def format_sci(v):
        """Format the value in scientific notation only for very small or very large numbers"""
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
        else:
            return f"{v:.2f}".rstrip('0').rstrip('.')  # Remove trailing zeros and dots if needed
    if heatmap.collections and heatmap.collections[0].colorbar:
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=font_scale*12)  # Adjust colorbar font size
    
    # Format labels in scientific notation
    x_labels_sci = [format_sci(v) for v in x_values]
    y_labels_sci = [format_sci(v) for v in y_values]

    ax.set_xticks(np.arange(len(x_values)) + 0.5)  
    ax.set_xticklabels(rf"{x_labels_sci}", rotation=45, ha="right", fontsize=12*font_scale)

    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels(rf"{y_labels_sci}", fontsize=12*font_scale)

    ax.tick_params(axis='both', labelsize=12*font_scale)

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
                      log_scale=False, ignored_keys=None, max_ = False, overall=True,
                      invert_x=False, invert_y=False, sub_plot=True, font_scale=1):
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
    # print(f"Plotting heatmaps for {param_x} and {param_y} with result metric: {result_metric}")
    if max_:
        print("Maximizing the result metric")
        # override behavior: just call best-fixed version
        plot_heatmap_best_fixed(
            results,
            param_x,
            param_y,
            result_metric,
            save_path=save_path,
            invert_colors=invert_colors,
            log_scale=log_scale,
            ignored_keys=ignored_keys,
            overall=overall,
            invert_x=invert_x,
            invert_y=invert_y,
            fig_size=fig_size,
            font_scale=font_scale
        )
        return
    # print("Finding fixed indices...")
    all_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)
    # print(f"Found {len(all_indices)} fixed indices.")
    plot_multiple_heatmaps(results, param_x, param_y, result_metric, all_indices,
                           fig_size, save_path, invert_colors, log_scale, ignored_keys=ignored_keys, 
                           invert_x=invert_x, invert_y=invert_y, sub_plot=sub_plot, font_scale=font_scale)
   



# plot multiple heatmaps in a grid with the same color scale
def plot_multiple_heatmaps(results, param_x, param_y, result_metric,
                           fixed_indices=None, fig_size=(12, 10), save_path="",
                           invert_colors=False, log_scale=False, ignored_keys=None,
                           invert_x=False, invert_y=False, sub_plot=True, font_scale=1):

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
    # print(f"Plotting heatmaps for {param_x} and {param_y} with result metric: {result_metric}")
    
    ignored_keys = ignored_keys or []
    
    if fixed_indices is None:
        fixed_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)

    if len(fixed_indices) == 1:
        plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_indices[0],
                           save_path=save_path, log_scale=log_scale, invert_colors=invert_colors, ignored_keys=ignored_keys, invert_x=invert_x, invert_y=invert_y, font_scale=font_scale)
        return
    num_rows = len(fixed_indices) // 2 + (len(fixed_indices) % 2)  # Arrange subplots in a 2-column layout
    num_cols = 2  # Maximum 2 heatmaps per row

    if sub_plot:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size, constrained_layout=True)
        axes = axes.flatten()
    else:
        axes = [None] * len(fixed_indices)

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
        if invert_x:
            x_values = list(reversed(x_values))
        # print(f"Y-axis not inverted: {y_values}")
        if invert_y:
            y_values = list(reversed(y_values))
            # print(f"Y-axis inverted: {y_values}")


        heatmap_matrix = np.zeros((len(y_values), len(x_values)))
        annot_matrix = np.empty_like(heatmap_matrix, dtype=object)

        for (x, y), (mean_val, err_val) in data.items():
            x_idx = x_values.index(x)
            y_idx = y_values.index(y)
            heatmap_matrix[y_idx, x_idx] = mean_val
            annot_matrix[y_idx, x_idx] = f"{mean_val:.3f}\n±{err_val:.3f}" if err_val > 0 else f"{mean_val:.3f}"

        def format_sci(v):
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
                return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
            else:
                return f"{v:.2f}".rstrip('0').rstrip('.')

        x_labels_sci = [rf"{format_sci(v)}" for v in x_values]
        y_labels_sci = [rf"{format_sci(v)}" for v in y_values]
        cmap = "coolwarm_r" if invert_colors else "coolwarm"

        # Create axis if in individual mode
        if not sub_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            ax = axes[i]

        heatmap = sns.heatmap(
            heatmap_matrix, xticklabels=x_labels_sci, yticklabels=y_labels_sci,
            annot=annot_matrix, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax,
            norm=norm, fmt="", annot_kws={"size": 10*font_scale}, cbar=True
        )
        if heatmap.collections and heatmap.collections[0].colorbar:
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=font_scale*12)

        varying_params_str = ", ".join(f"{format_display_name(key)}={fixed_params[key]}" for key in varying_keys)
        ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=12*font_scale)
        ax.set_ylabel(rf"{format_display_name(param_y)}", fontsize=12*font_scale)
        ax.tick_params(axis='both', labelsize=12*font_scale)
        ax.set_title(rf"Heatmap with parameters:\n{varying_params_str}", fontsize=14*font_scale)

        if not sub_plot and save_path:
            suffix = "_".join(f"{key}_{fixed_params[key]}" for key in varying_keys)
            filename = f"{save_path}_{suffix}.png"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"Saved heatmap as {filename}")
            plt.close(fig)
    # Remove unused axes if any (only in subplot mode)
    if sub_plot and len(fixed_indices) < len(axes):
        for j in range(len(fixed_indices), len(axes)):
            fig.delaxes(axes[j])

    if sub_plot:
        if save_path:
            plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
            print(f"Saved combined subplot figure as {save_path}.png")
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
                         title=rf"3D Scatter of {format_display_name(result_metric)} by {format_display_name(param_x)}, {format_display_name(param_y)}, and {format_display_name(param_z)}",
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


def smart_formatter(val):
    if val == 0:
        return "0"
    abs_val = abs(val)
    if 1e-2 <= abs_val < 1e3:
        return f"{val:,.2f}".replace(",", " ").replace(".", ",").rstrip('0').rstrip(',')
    else:
        exponent = int(np.floor(np.log10(abs_val)))
        base = round(val / (10 ** exponent), 1)
        if base == 1.0:
            return f"$10^{{{exponent}}}$"
        else:
            return f"${base}\\times10^{{{exponent}}}$"

def format_ticks_smart(axis, axis_type='x'):
    formatter = mticker.FuncFormatter(lambda val, _: smart_formatter(val))
    if axis_type == 'x':
        axis.xaxis.set_major_formatter(formatter)
    else:
        axis.yaxis.set_major_formatter(formatter)



def assign_gradient_colors(sorted_keys, cmap_name='viridis'):
    """
    Assign colors from a colormap based on sorted keys.
    """
    cmap = get_cmap(cmap_name)
    num_keys = len(sorted_keys)
    return {key: cmap(i / max(1, num_keys - 1)) for i, key in enumerate(sorted_keys)}


def plot_metrics_vs_param(results, param_x, metrics, group_by=None,
                          split_by=None, title="", grid=True, save_path=None,
                          ylim=None, log_scale_x=False, log_scale_y=False,
                          sub_plot=True, max_overall=False, show_plot=True, use_color_gradient=True,
                          font_scale=1.0, GT_plot=True, stds=None):
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
    if isinstance(metrics, str):
        metrics = [metrics]
    

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
                            title, grid, ylim, log_scale_x, log_scale_y, max_overall=max_overall, 
                            use_color_gradient=use_color_gradient, font_scale=font_scale, GT_plot=GT_plot, stds=stds)
            
            
            format_ticks_smart(ax, 'x')
            format_ticks_smart(ax, 'y')

        # Hide unused subplots
        for j in range(num_plots, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        plt.tight_layout()
        if save_path:
            full_path = f"{save_path}.png"
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            print(f"Saved combined subplot figure to: {full_path}")
        if show_plot:
            plt.show()

    else:
        # === SEPARATE FIGURES MODE ===
        for split_key, group_results in split_groups.items():
            fig, ax = plt.subplots(figsize=(9, 6))
            _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                            metric_styles, plt.cm.tab10(np.linspace(0, 1, 10)),
                            split_key, split_by, title, grid, ylim, log_scale_x, log_scale_y, 
                            use_color_gradient=use_color_gradient, max_overall=max_overall, font_scale=font_scale, GT_plot=GT_plot)
            
            format_ticks_smart(ax, 'x')

            format_ticks_smart(ax, 'y')


            plt.tight_layout()
            if save_path:
                suffix = "_".join(f"{k}_{v}" for k, v in split_key)
                full_path = f"{save_path}_{suffix}.png"
                plt.savefig(full_path, bbox_inches="tight", dpi=300)
                print(f"Saved individual plot to: {full_path}")
            if show_plot:
                plt.show()


def _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                    metric_styles, color_cycle, split_key, split_by,
                    title, grid, ylim, log_scale_x, log_scale_y,
                    max_overall=False, use_color_gradient=False, font_scale=1.0, GT_plot=True, stds = None):
    """
    Internal helper: plots one panel (subplot or full figure).
    If max_overall=True, selects the best value across all other hyperparameters
    (except param_x, group_by, split_by).
    """

    grouped = defaultdict(list)
    for exp in group_results:
        group_key = tuple((k, exp['params'].get(k, None)) for k in group_by)
        grouped[group_key].append(exp)

    sorted_keys = sorted(grouped.keys())
    color_map = assign_gradient_colors(sorted_keys) if use_color_gradient else {
        group: color_cycle[i % len(color_cycle)] for i, group in enumerate(sorted_keys)
    }

    for group_key in sorted_keys:
        exps = grouped[group_key]
        grouped_by_x = defaultdict(list)
        for exp in exps:
            x = exp['params'][param_x]
            grouped_by_x[x].append(exp)

        x_vals = sorted(grouped_by_x.keys())
        metric_vals = {metric: [] for metric in metrics}
        metric_errs = {metric: [] for metric in metrics}

        for x in x_vals:
            exp_list = grouped_by_x[x]
            for metric in metrics:
                best_mean = None
                best_err = None
                for exp in exp_list:
                    values = exp['results'][metric]
                    # Handle metrics that are single float vs list of floats vs list of lists
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], list):  # list of lists → take last value of each
                            values = [v[-1] for v in values]
                        # else: list of floats → use as is
                    elif isinstance(values, float):  # single float → wrap
                        values = [values]

                    mean_val = np.mean(values)
                    if stds is not None:
                        err_val = np.mean(exp['results'][stds])
                    else:
                        err_val = sem(values) if len(values) > 1 else 0.0
                    is_loss = "loss" in metric.lower() or "error" in metric.lower()
                    if best_mean is None or (is_loss and mean_val < best_mean) or (not is_loss and mean_val > best_mean):
                        best_mean = mean_val
                        best_err = err_val
                if max_overall:
                    metric_vals[metric].append(best_mean)
                    metric_errs[metric].append(best_err)
                else:
                    all_means = []
                    all_errs = []
                    for exp in exp_list:
                        values = exp['results'][metric]
                        if isinstance(values, list) and len(values) > 0:
                            if isinstance(values[0], list):  # list of lists → take last value of each
                                values = [v[-1] for v in values]
                            # else: list of floats → use as is
                        elif isinstance(values, float):  # single float → wrap
                            values = [values]
                        all_means.append(np.mean(values))
                        all_errs.append(sem(values) if len(values) > 1 else 0.0)
                    metric_vals[metric].append(np.mean(all_means))
                    metric_errs[metric].append(np.mean(all_errs))

        for metric in metrics:
            style = metric_styles[metric]
            # Label intelligent : si une seule métrique (ex. gt_accuracy), on affiche juste le group_by
            if len(metrics) == 1:
                metric_name = format_display_name(metric)
                label = ", ".join(f"{format_display_name(k)}={v}" for k, v in group_key) if group_by else metric_name
            else:
                # Plusieurs métriques → on précise le nom
                label = f"{format_display_name(metric)} ({', '.join(f'{format_display_name(k)}={v}' for k, v in group_key)})" if group_by else format_display_name(metric)

            yerrs = np.array(metric_errs[metric])
            if np.any(yerrs > 0):
                # Erreurs non nulles → errorbar
                ax.errorbar(
                    x_vals,
                    metric_vals[metric],
                    yerr=metric_errs[metric],
                    fmt=style['marker'] + style['linestyle'],
                    capsize=5,
                    label=rf"{label}",
                    color=color_map[group_key]
                )
            else:
                # Pas d’erreur → simple courbe
                ax.plot(
                    x_vals,
                    metric_vals[metric],
                    style['marker'] + style['linestyle'],
                    label=rf"{label}",
                    color=color_map[group_key]
                )


    split_label = ", ".join(f"{format_display_name(k)}={v}" for k, v in split_key) if split_by else ""
    full_title = f"{title}\n{split_label}" if split_label else title
    ax.set_title(rf"{full_title}", fontsize=14 * font_scale)
    ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=12 * font_scale)
    ax.set_ylabel(
        rf"{', '.join(format_display_name(m) for m in metrics)}" if len(metrics) > 1 
        else rf"{format_display_name(metrics[0])}",
        fontsize=12 * font_scale
    )

    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    if ylim:
        ax.set_ylim(ylim)
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=11 * font_scale)
    # === Ajouter Ground Truth Accuracy (courbe grise) si pertinent ===
    if (metrics == ["accuracy"]) and GT_plot:
        # Trouve la plus grande valeur de K
        K_vals = [exp['params'].get('K') for exp in group_results if 'K' in exp['params']]
        if K_vals:
            max_K = max(K_vals)
            gt_x = []
            gt_y = []
            for x in x_vals:
                matching_exps = [exp for exp in grouped_by_x[x] if exp['params'].get('K') == max_K]
                if matching_exps:
                    gt_avg = np.mean([np.mean(exp['results']['gt_accuracy']) for exp in matching_exps if 'gt_accuracy' in exp['results']])
                    gt_x.append(x)
                    gt_y.append(gt_avg)

            if gt_x and gt_y:
                ax.plot(gt_x, gt_y, linestyle="--", color="gray", label=r"GT")
    ax.legend(fontsize=11 * font_scale)    


