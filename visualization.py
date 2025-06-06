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
# Visualization Utilities for Experimental Results
#
# This module contains functions to:
# - Plot performance metrics across parameter sweeps (e.g., accuracy, reconstruction error)
# - Visualize grouped results for comparisons across strategies or settings
# - Extract and display the best-performing hyperparameter configurations
# - Generate plots compatible with scientific reporting (e.g., heatmaps, grouped lines)
############################################


def format_display_name(name):
    """
    Converts internal metric or parameter names into clean, human-readable labels
    suitable for plots, legends, and reports.

    Common conversions include:
    - Replacing underscores with spaces and title-casing.
    - Mapping technical terms to LaTeX-friendly or descriptive equivalents.

    Examples:
    - "gt_log_likelihoods" → "GT Log Likelihood"
    - "val_losses" → "Validation Loss"
    - "weight_decay" → "Weight Decay"
    - "pxK" → "$p \\cdot k$"

    Parameters:
    - name (str): Internal name used in experiment results or parameters.

    Returns:
    - str: Formatted label for display.
    """

    name_map = {
        # === Metric display names ===
        "train_losses": "Training Loss",
        "val_losses": "Validation Loss",
        "accuracy": "Accuracy",
        "log_likelihoods": "Log Likelihood",
        "gt_accuracy": "GT Accuracy",
        "gt_log_likelihoods": "GT Log Likelihood",
        "reconstruction_errors": "Reconstruction Error",
        "reconstruction_error_scaled": "Reconstruction Error (Scaled)",
        "svd_error_scaled": "SVD Error (Scaled)",
        "gt_loss": "GT Loss",
        "pearson_corr": "Pearson Correlation",
        "spearman_corr": "Spearman Correlation",

        # === Parameter display names ===
        "lr": "Learning Rate",
        "weight_decay": "Weight Decay",
        "num_epochs": "Num Epochs",
        "num_data_points": "Num Data Points",
        "p": "$p$",
        "d": "Embedding Dim ($d$)",
        "d1": "Init Dim (d1)",
        "K": "$k$",
        "n": "$n$",
        "m": "^$m$",
        "s": "$s$",
        "alpha": "$\\alpha(s)$",
        "pxK": "$p \\cdot k$",
        "norm_ratio": "$\\|UV^T\\|/\\|X^*\\|$",
        "norm_ratio_scaled": "$\\|\\alpha(s) UV^T\\|/\\|X^*\\|$",

        # === Strategy or labeling categories ===
        "strategy": "Strat",
        "popularity": "Popularity",
        "cluster": "Cluster",
        "proximity": "Max-Min",
        "svd": "SVD",
        "top_k": "Top 10\\%",
        "p*s": "p$\\cdot$s",
        "margin": "Close-Call",
        "variance": "high $\\sigma$",
    }

    # Return mapped name if available, otherwise convert snake_case → Title Case
    if name in name_map:
        return name_map[name]
    else:
        return name.replace("_", " ").title()

def plot_losses(results, param_index=None, selected_indices=None, save_path=""):
    """
    Visualizes training and validation losses from a list of experiments.

    Supports two display modes:
    - Single experiment (with param_index): Plots both train and val losses on one graph.
    - Multiple experiments: Plots selected experiments' losses separately with color-coded labels.

    Features:
    - Displays only the last repetition for each experiment.
    - Highlights only the parameters that vary across experiments.
    - Labels are neatly arranged in 4 columns below the plot.
    - Optional saving of plots to PNG files.

    Parameters:
    - results (list): Output list from `parameter_scan`, containing metrics and parameters.
    - param_index (int, optional): If provided, show a detailed loss curve for a single experiment.
    - selected_indices (list, optional): Indices of experiments to display. Defaults to all.
    - save_path (str, optional): Base filename to save plots. If empty, plots are not saved.
    """

    def format_params(params):
        """Formats parameter dictionary into a readable string for plot titles."""
        return ", ".join(f"{format_display_name(key)}: {value}" for key, value in params.items())

    def find_varying_params(results):
        """Identifies which parameters vary across the list of experiments."""
        all_keys = results[0]['params'].keys()
        varying_params = {key for key in all_keys if len(set(exp['params'][key] for exp in results)) > 1}
        return list(varying_params)

    # === Case 1: Single experiment plot (Train & Validation losses) ===
    if param_index is not None:
        exp = results[param_index]
        formatted_params = format_params(exp['params'])

        plt.figure(figsize=(10, 5))
        plt.plot(exp['results']['train_losses'][-1], label='Training Loss', linestyle='--')
        plt.plot(exp['results']['val_losses'][-1], label='Validation Loss')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(rf"Train \& Val Loss for\\{formatted_params}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        if save_path:
            plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)

        plt.show()

    # === Case 2: Multiple experiment curves with label comparison ===
    else:
        varying_params = find_varying_params(results)
        param_names = ", ".join(format_display_name(p) for p in varying_params)

        # Select which experiments to plot
        if selected_indices is None:
            selected_indices = range(len(results))

        # Assign unique color per experiment
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

        # Build formatted label strings for each experiment
        param_texts = []
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            param_values = ", ".join(f"{format_display_name(key)}={exp['params'][key]}" for key in varying_params)
            param_texts.append((colors[i], f"Exp {exp_idx+1}: {param_values}"))

        def display_labels():
            """Displays labels in 4 columns under the plot with matching curve colors."""
            num_cols = 4
            x_positions = [0.02, 0.27, 0.52, 0.77]
            for i, (color, text) in enumerate(param_texts):
                row = i // num_cols
                col = i % num_cols
                y_position = -0.07 - row * 0.05
                plt.figtext(x_positions[col], y_position, rf"$\text{{{text}}}$", color=color, fontsize=9, ha="left")

        # === Plot all training losses ===
        plt.figure(figsize=(10, 5))
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            plt.plot(exp['results']['train_losses'][-1], color=colors[i])

        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title(rf"Losses for the parameter scan of the variables:\\ {param_names}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        display_labels()

        if save_path:
            plt.savefig(f"{save_path}_train.png", bbox_inches="tight", dpi=300)

        plt.show()

        # === Plot all validation losses ===
        plt.figure(figsize=(10, 5))
        for i, exp_idx in enumerate(selected_indices):
            exp = results[exp_idx]
            plt.plot(exp['results']['val_losses'][-1], color=colors[i])

        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title(rf"Losses for the parameter scan of the variables:\\ {param_names}", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        display_labels()

        if save_path:
            plt.savefig(f"{save_path}_val.png", bbox_inches="tight", dpi=300)

        plt.show()

def plot_heatmap_best_fixed(results, param_x, param_y, result_metric, save_path="", invert_colors=False, log_scale=False, ignored_keys=None, overall=True, invert_x=False, invert_y=False, fig_size=(10, 7), font_scale=1):
    """
    Plots a heatmap to visualize the mean value (± SEM) of a metric across 2 parameters.

    Two modes:
    - overall=True: Filters experiments to match the best global configuration (except param_x and param_y).
    - overall=False: Considers all experiments and selects the best result per (param_x, param_y) pair.

    Parameters:
    - results (list): Output from `parameter_scan`, containing parameter configurations and results.
    - param_x (str): Name of the parameter for the x-axis.
    - param_y (str): Name of the parameter for the y-axis.
    - result_metric (str): Metric to visualize (e.g., "accuracy", "reconstruction_errors").
    - save_path (str): If specified, saves the plot as a PNG file.
    - invert_colors (bool): If True, reverses the colormap (e.g., red = best).
    - log_scale (bool): Whether to apply logarithmic color normalization.
    - ignored_keys (list): Parameters to exclude from the "fixed" filtering logic (used in overall=True mode).
    - overall (bool): Toggle between global best filtering and full aggregation.
    - invert_x (bool): Reverse the order of x-axis values.
    - invert_y (bool): Reverse the order of y-axis values.
    - fig_size (tuple): Figure size in inches.
    - font_scale (float): Scaling factor for all text in the plot.
    """
    
    ignored_keys = ignored_keys or []
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()
    data = {}

    # === Mode 1: Only use configs that match the best overall config (excluding param_x and param_y) ===
    if not overall:
        best_exp_index = min(range(len(results)), key=lambda i: min(results[i]['results'][result_metric])) if is_loss else \
                         max(range(len(results)), key=lambda i: max(results[i]['results'][result_metric]))
        best_params = results[best_exp_index]['params']

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

    # === Mode 2: Consider all configurations and pick best mean value per (x, y) ===
    else:
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

    # === Prepare matrix for heatmap ===
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
        annot_matrix[yi, xi] = rf"${mean_val:.4f}\\\pm{err_val:.4f}$" if err_val > 0 else rf"${mean_val:.4f}$"

    # === Set color normalization ===
    vmin = np.percentile([v[0] for v in data.values()], 5)
    vmax = np.percentile([v[0] for v in data.values()], 95)
    if log_scale:
        vmin = max(vmin, 1e-5)
        vmax = max(vmax, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # === Plot heatmap ===
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = "coolwarm_r" if invert_colors else "coolwarm"
    heatmap = sns.heatmap(
        heatmap_matrix, annot=annot_matrix, fmt="", cmap=cmap,
        xticklabels=[], yticklabels=[], norm=norm,
        annot_kws={"size": 12*font_scale}, ax=ax, cbar=True
    )

    # Resize colorbar text if available
    if heatmap.collections and heatmap.collections[0].colorbar:
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=font_scale*12)

    def format_sci(v):
        """Formats float using scientific notation for small or large values."""
        return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-") if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0) else f"{v:.2f}".rstrip('0').rstrip('.')

    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels([format_sci(v) for v in x_values], rotation=45, ha="right", fontsize=12*font_scale)
    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels([format_sci(v) for v in y_values], fontsize=12*font_scale)

    ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=14*font_scale)
    ax.set_ylabel(rf"{format_display_name(param_y)}", fontsize=14*font_scale)
    ax.set_title(
        rf"Heatmap of {format_display_name(result_metric)} by {format_display_name(param_x)} and {format_display_name(param_y)} "
        f"({'global best block' if overall else 'best per (x,y)'})",
        fontsize=16*font_scale
    )
    ax.tick_params(axis='both', labelsize=12*font_scale)

    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    plt.show()

def enrich_params_with_data_points(results):
    """
    Adds a new derived parameter 'num_data_points' to each experiment's config.

    This represents the total number of training samples used in each experiment,
    calculated as:
        num_data_points = n * m * p * 0.5 * K

    This is useful for visualizing or grouping results based on the effective dataset size.

    Notes:
    - The factor 0.5 accounts for symmetric triplets (e.g., (u, i, j) and (u, j, i)).
    - Values are rounded to 4 decimals to avoid floating point comparison issues.

    Parameters:
    - results (list): List of experiment dictionaries as returned by `parameter_scan`.

    Returns:
    - list: Same list with each experiment's 'params' updated to include 'num_data_points'.
    """
    for exp in results:
        n = exp['params']['n']
        m = exp['params']['m']
        p = exp['params']['p']
        exp['params']['num_data_points'] = round(n * m * p * 0.5, 4)

    return results


# Plot the heatmap for two selected parameters with a result metric,
# keeping all other parameters fixed (as specified by a selected configuration).
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index,
                        save_path="", invert_colors=False, log_scale=False,
                        ignored_keys=None, overall=True,
                        invert_x=False, invert_y=False, ax=None, font_scale=1):
    """
    Plots a heatmap of two chosen parameters against a selected result metric.
    All other parameters are fixed according to a selected experiment (via `fixed_index`).

    This is useful for comparing the effect of two hyperparameters while holding others constant.

    Parameters:
    - results (list): The list of experiment outputs from parameter_scan.
    - param_x (str): Name of the parameter to use on the X-axis.
    - param_y (str): Name of the parameter to use on the Y-axis.
    - result_metric (str): The performance metric to visualize (e.g., accuracy, loss).
    - fixed_index (int): Index of the experiment used to fix all other hyperparameters.
    - save_path (str, optional): Path to save the figure (if ax is None).
    - invert_colors (bool, optional): If True, reverse the colormap direction.
    - log_scale (bool, optional): If True, apply logarithmic normalization to the color scale.
    - ignored_keys (list, optional): Parameters to ignore when comparing for fixed settings.
    - invert_x (bool, optional): If True, reverse the order of the x-axis values.
    - invert_y (bool, optional): If True, reverse the order of the y-axis values.
    - ax (matplotlib.axes.Axes, optional): If provided, plot into existing axes. Otherwise, a new figure is created.
    - font_scale (float, optional): Scaling factor for all font sizes in the plot.
    """
    fixed_params = results[fixed_index]['params']
    data = {}
    ignored_keys = ignored_keys or []

    # === Step 1: Select experiments that match the fixed configuration ===
    for exp in results:
        if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y] + ignored_keys):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            values = exp['results'][result_metric]
            mean_val = np.mean(values)
            err_val = sem(values) if len(values) > 1 else 0  # Compute error bar
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = (mean_val, err_val)
            else:
                # Average values if multiple experiments exist for the same config
                prev_mean, prev_err = data[(x_val, y_val)]
                data[(x_val, y_val)] = ((prev_mean + mean_val)/2, (prev_err + err_val)/2)

    # === Step 2: Prepare axis values and optionally reverse order ===
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    if invert_x:
        x_values = list(reversed(x_values))
    if invert_y:
        y_values = list(reversed(y_values))

    # === Step 3: Build the heatmap matrix and annotation matrix ===
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    annot_matrix = np.empty_like(heatmap_matrix, dtype=object)
    for (x, y), (mean_val, err_val) in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = mean_val
        if err_val > 0:
            annot_matrix[y_idx, x_idx] = rf"${mean_val:.4f}\\\pm\,{err_val:.4f}$"
        else:
            annot_matrix[y_idx, x_idx] = rf"${mean_val:.4f}$"

    # === Step 4: Create or use existing plot axes ===
    standalone_mode = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        standalone_mode = True

    # === Step 5: Color normalization for better visualization ===
    mean_vals = [mean for (mean, _) in data.values()]
    vmin, vmax = np.percentile(mean_vals, [5, 95])
    if log_scale:
        vmin = max(vmin, 1e-5)
        vmax = max(vmax, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # === Step 6: Generate the heatmap ===
    cmap = "coolwarm_r" if invert_colors else "coolwarm"
    heatmap = sns.heatmap(heatmap_matrix, cmap=cmap, norm=norm, ax=ax,
                          annot=annot_matrix, fmt='', annot_kws={"size": 12 * font_scale}, cbar=True)

    # === Step 7: Axis formatting and labels ===
    ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=14 * font_scale)
    ax.set_ylabel(rf"{format_display_name(param_y)}", fontsize=14 * font_scale)

    def format_sci(v):
        """Format numeric values into scientific notation when appropriate."""
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
        else:
            return f"{v:.2f}".rstrip('0').rstrip('.')

    # Format ticks and apply scientific formatting
    x_labels_sci = [format_sci(v) for v in x_values]
    y_labels_sci = [format_sci(v) for v in y_values]
    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_xticklabels([rf"${v}$" for v in x_labels_sci], rotation=45, ha="right", fontsize=12 * font_scale)
    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_yticklabels([rf"${v}$" for v in y_labels_sci], fontsize=12 * font_scale)
    ax.tick_params(axis='both', labelsize=12 * font_scale)

    # === Step 8: Save or display plot ===
    if standalone_mode and save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved heatmap as {save_path}.png")

    if standalone_mode:
        plt.show()



def find_fixed_indices(results, param_x, param_y, ignored_keys=None):
    """
    Identifies the indices of experiments corresponding to unique settings of fixed parameters.

    This function is used to group experiments based on shared values of all parameters 
    except the two selected for variation (param_x and param_y), and any keys explicitly ignored.

    Parameters:
    - results (list): List of experiment dictionaries, typically output from parameter_scan.
    - param_x (str): Name of the parameter varying along the x-axis.
    - param_y (str): Name of the parameter varying along the y-axis.
    - ignored_keys (list, optional): List of parameter names to ignore when determining fixed configurations.

    Returns:
    - list[int]: Indices in `results` that correspond to the first occurrence of each distinct 
                 fixed configuration (excluding `param_x`, `param_y`, and any ignored keys).
    """
    # Initialize the list of ignored keys if not provided
    ignored_keys = ignored_keys or []

    # Dictionary to store already seen fixed parameter combinations
    unique_fixed_params = {}

    # List to store the first index where each fixed parameter combination appears
    fixed_indices = []

    # Iterate over each experiment in the result list
    for idx, exp in enumerate(results):
        # Create a tuple of all parameters except param_x, param_y, and ignored keys
        fixed_params = tuple(
            (k, v) for k, v in exp['params'].items()
            if k not in [param_x, param_y] + ignored_keys
        )

        # If this combination has not been seen before, save its index
        if fixed_params not in unique_fixed_params:
            unique_fixed_params[fixed_params] = idx
            fixed_indices.append(idx)

    return fixed_indices

# Plot multiple heatmaps, one for each fixed configuration of remaining parameters
def plot_all_heatmaps(results, param_x, param_y, result_metric,
                      fig_size=(12, 10), save_path="", invert_colors=False,
                      log_scale=False, ignored_keys=None, max_ = False, overall=True,
                      invert_x=False, invert_y=False, sub_plot=True, font_scale=1):
    """
    Plots all heatmaps in a grid with the same color scale.

    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter (x-axis).
    - param_y (str): Name of the second parameter (y-axis).
    - result_metric (str): The result metric to use for coloring the heatmap.
    - fig_size (tuple): Figure size (width, height).
    - save_path (str): Path to save the figure. If empty, plot is not saved.
    - invert_colors (bool): Reverse the colormap if True.
    - log_scale (bool): Apply log scale to the heatmap color values.
    - ignored_keys (list): Parameters to ignore when determining fixed configs.
    - max_ (bool): If True, uses only the best-performing config (global max or min).
    - overall (bool): Passed to plot_heatmap_best_fixed to control filtering logic.
    - invert_x (bool): If True, reverse the order of the x-axis.
    - invert_y (bool): If True, reverse the order of the y-axis.
    - sub_plot (bool): Whether to use subplots for multiple heatmaps.
    - font_scale (float): Global scale factor for all font sizes.
    """
    
    # Special case: show only the heatmap for the overall best configuration
    if max_:
        print("Maximizing the result metric")
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
        return  # Exit after plotting the best config

    # Find indices of experiments with unique fixed parameter configurations
    all_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)

    # Plot a heatmap for each unique fixed configuration
    plot_multiple_heatmaps(results, param_x, param_y, result_metric, all_indices,
                           fig_size, save_path, invert_colors, log_scale, ignored_keys=ignored_keys, 
                           invert_x=invert_x, invert_y=invert_y, sub_plot=sub_plot, font_scale=font_scale)



# Plot multiple heatmaps arranged in a grid, each corresponding to a fixed configuration
def plot_multiple_heatmaps(results, param_x, param_y, result_metric,
                           fixed_indices=None, fig_size=(12, 10), save_path="",
                           invert_colors=False, log_scale=False, ignored_keys=None,
                           invert_x=False, invert_y=False, sub_plot=True, font_scale=1):
    """
    Plots multiple heatmaps in a grid with the same color scale.

    Parameters:
    - results (list): Output from parameter_scan.
    - param_x (str): Name of the x-axis parameter.
    - param_y (str): Name of the y-axis parameter.
    - result_metric (str): The metric to visualize (e.g. accuracy, loss).
    - fixed_indices (list): Indices corresponding to different fixed configurations.
    - fig_size (tuple): Overall size of the figure (when subplots are used).
    - save_path (str): If set, saves the plots to file.
    - invert_colors (bool): Reverses the colormap.
    - log_scale (bool): Applies log scale to the color axis.
    - ignored_keys (list): Keys to exclude when checking for fixed parameters.
    - invert_x (bool): Reverse order of x-axis values.
    - invert_y (bool): Reverse order of y-axis values.
    - sub_plot (bool): Whether to organize all plots in one figure (subplot mode).
    - font_scale (float): Controls all font sizes in the plot.
    """
    ignored_keys = ignored_keys or []

    # Determine which fixed configurations to use
    if fixed_indices is None:
        fixed_indices = find_fixed_indices(results, param_x, param_y, ignored_keys=ignored_keys)

    # Handle single heatmap directly
    if len(fixed_indices) == 1:
        plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_indices[0],
                           save_path=save_path, log_scale=log_scale,
                           invert_colors=invert_colors, ignored_keys=ignored_keys,
                           invert_x=invert_x, invert_y=invert_y, font_scale=font_scale)
        return

    # Determine grid layout
    num_rows = len(fixed_indices) // 2 + (len(fixed_indices) % 2)
    num_cols = 2

    # Create subplots if required
    if sub_plot:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size, constrained_layout=True)
        axes = axes.flatten()
    else:
        axes = [None] * len(fixed_indices)

    # Identify which parameters vary across all fixed configurations
    fixed_params_list = [results[idx]['params'] for idx in fixed_indices]
    all_keys = set(fixed_params_list[0].keys())
    constant_keys = {key for key in all_keys if all(d[key] == fixed_params_list[0][key] for d in fixed_params_list)}
    varying_keys = all_keys - constant_keys

    # Gather global min/max to normalize colormap scale across plots
    all_min_values = []
    all_max_values = []
    for fixed_index in fixed_indices:
        fixed_params = results[fixed_index]['params']
        for exp in results:
            if all(exp['params'][key] == fixed_params[key] for key in fixed_params if key not in [param_x, param_y]):
                values = exp['results'][result_metric]
                all_min_values.append(min(values))
                all_max_values.append(max(values))

    # Define color normalization
    vmin = np.percentile(all_min_values, 5)
    vmax = np.percentile(all_max_values, 95)
    if log_scale:
        vmin = max(vmin, 1e-5)
        vmax = max(vmax, vmin * 10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Loop through each heatmap to generate
    for i, fixed_index in enumerate(fixed_indices):
        fixed_params = results[fixed_index]['params']
        data = {}

        # Filter relevant experiments that match the fixed parameters
        for exp in results:
            if all(exp['params'].get(key) == fixed_params[key]
                   for key in fixed_params if key not in [param_x, param_y] + ignored_keys):
                x_val = exp['params'][param_x]
                y_val = exp['params'][param_y]
                values = exp['results'][result_metric]
                mean_val = np.mean(values)
                err_val = sem(values) if len(values) > 1 else 0.0
                data[(x_val, y_val)] = (mean_val, err_val)

        # Create axes
        x_values = sorted(set(k[0] for k in data.keys()))
        y_values = sorted(set(k[1] for k in data.keys()))
        if invert_x:
            x_values = list(reversed(x_values))
        if invert_y:
            y_values = list(reversed(y_values))

        # Build matrix for heatmap values and annotations
        heatmap_matrix = np.zeros((len(y_values), len(x_values)))
        annot_matrix = np.empty_like(heatmap_matrix, dtype=object)
        for (x, y), (mean_val, err_val) in data.items():
            x_idx = x_values.index(x)
            y_idx = y_values.index(y)
            heatmap_matrix[y_idx, x_idx] = mean_val
            annot_matrix[y_idx, x_idx] = rf"${mean_val:.3f}\\\pm{err_val:.3f}$" if err_val > 0 else rf"${mean_val:.3f}$"

        # Format numbers with scientific notation if needed
        def format_sci(v):
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
                return f"{v:.1e}".replace("e+00", "").replace("e+0", "e").replace("e-0", "e-")
            else:
                return f"{v:.2f}".rstrip('0').rstrip('.')

        x_labels_sci = [rf"{format_sci(v)}" for v in x_values]
        y_labels_sci = [rf"{format_sci(v)}" for v in y_values]
        cmap = "coolwarm_r" if invert_colors else "coolwarm"

        # Prepare axes for current subplot
        if not sub_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            ax = axes[i]

        # Plot heatmap
        heatmap = sns.heatmap(
            heatmap_matrix, xticklabels=x_labels_sci, yticklabels=y_labels_sci,
            annot=annot_matrix, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax,
            norm=norm, fmt="", annot_kws={"size": 10*font_scale}, cbar=True
        )

        if heatmap.collections and heatmap.collections[0].colorbar:
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=font_scale*12)

        # Add plot titles and axis labels
        varying_params_str = ", ".join(f"{format_display_name(key)}={fixed_params[key]}" for key in varying_keys)
        ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=12*font_scale)
        ax.set_ylabel(rf"{format_display_name(param_y)}", fontsize=12*font_scale)
        ax.tick_params(axis='both', labelsize=12*font_scale)
        ax.set_title(rf"Heatmap with parameters:\\{varying_params_str}", fontsize=14*font_scale)

        # Save individual plots if not using subplots
        if not sub_plot and save_path:
            suffix = "_".join(f"{key}_{fixed_params[key]}" for key in varying_keys)
            filename = f"{save_path}_{suffix}.png"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"Saved heatmap as {filename}")
            plt.close(fig)

    # Remove unused subplots if any
    if sub_plot and len(fixed_indices) < len(axes):
        for j in range(len(fixed_indices), len(axes)):
            fig.delaxes(axes[j])

    # Save combined figure and display
    if sub_plot:
        if save_path:
            plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
            print(f"Saved combined subplot figure as {save_path}.png")
        plt.show()


# Display a list of experiment indices and their corresponding parameter settings
def display_experiment_indices(results):
    """
    Displays the index and corresponding parameters for each experiment.
    
    Parameters:
    - results (list): The output from parameter_scan.
    """

    # Print table header
    print("\nAvailable Experiments:")
    print("Index | Parameters")
    print("--------------------------------------")

    # Iterate over each experiment and print its index and parameter set
    for idx, exp in enumerate(results):
        params = exp['params']
        # Convert parameter dictionary to readable string
        params_str = ', '.join(f"{key}={value}" for key, value in params.items())
        # Print the index and formatted parameters
        print(f"{idx:<5} | {params_str}")

    # Display note for user on how to use this output
    print("\nUse these indices to select experiments in other functions like plot_losses or plot_heatmap_fixed.")

# Plot a 3D scatter plot to visualize the effect of three parameters on a result metric
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
    
    # === Build a list of dictionaries with parameter values and result metric
    data = []
    for exp in results:
        data.append({
            param_x: exp['params'][param_x],
            param_y: exp['params'][param_y],
            param_z: exp['params'][param_z],
            result_metric: max(exp['results'][result_metric])  # Use the best score across repetitions
        })

    # === Convert list to a DataFrame for Plotly visualization
    df = pd.DataFrame(data)

    # === Create an interactive 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df,
        x=param_x, y=param_y, z=param_z, color=result_metric,
        title=rf"3D Scatter of {format_display_name(result_metric)} by {format_display_name(param_x)}, {format_display_name(param_y)}, and {format_display_name(param_z)}",
        labels={param_x: param_x, param_y: param_y, param_z: param_z, result_metric: result_metric},
        opacity=0.8
    )

    # === Display the interactive plot in the browser or notebook
    fig.show()

# Gets best hyperparameter configuration for a given result metric
def get_best_params(results, result_metric):
    """
    Finds the best hyperparameter configuration for a given result metric.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - result_metric (str): The result metric to optimize (e.g., accuracy, loss, error).
    
    Returns:
    - dict: The best hyperparameter configuration (from results[i]['params']).
    - int: The index of the best experiment in the results list.
    """

    # === Determine optimization direction: minimize for losses/errors, maximize otherwise
    is_loss = "loss" in result_metric.lower() or "error" in result_metric.lower()

    # === Select the experiment index with best score (min or max over all repetitions)
    best_exp_index = min(
        range(len(results)), 
        key=lambda i: min(results[i]['results'][result_metric])
    ) if is_loss else max(
        range(len(results)), 
        key=lambda i: max(results[i]['results'][result_metric])
    )

    # === Extract the best experiment and its parameters
    best_exp = results[best_exp_index]
    best_params = best_exp['params']
    best_value = min(best_exp['results'][result_metric]) if is_loss else max(best_exp['results'][result_metric])

    # === Display the result
    print(f"Best parameters for {result_metric} (Index: {best_exp_index}): {best_params}, Best value: {best_value}")

    return best_params, best_exp_index

# Gets best hyperparameter configuration for all result metrics
def get_best_params_all_metrics(results):
    """
    Finds the best hyperparameter configuration for each result metric in the dataset.
    
    Parameters:
    - results (list): The output from parameter_scan.
    
    Returns:
    - dict: A dictionary mapping each result metric to a tuple:
            (best hyperparameter configuration, index of best experiment).
    """

    # === Extract the list of available result metrics from the first experiment
    all_metrics = results[0]['results'].keys()
    best_params_per_metric = {}

    # === Iterate over each metric and compute the best config
    for metric in all_metrics:
        best_params_per_metric[metric] = get_best_params(results, metric)

    return best_params_per_metric

# Print the results for selected experiment indices and a specific metric (or all if not specified)
def print_results(results, indices=None, params_off=False, metric=None):
    """
    Prints the results of the experiments with optional filtering and display options.

    Parameters:
    - results (list): The output from parameter_scan.
    - indices (list, optional): List of experiment indices to display. If None, displays all experiments.
    - params_off (bool, optional): If True, hides the hyperparameter configurations in the output.
    - metric (str, optional): Specific result metric to display. If None, shows the first metric available.
    """

    # Use all experiment indices if not specified
    if indices is None:
        indices = range(len(results))

    # Use the first available metric if none is specified
    if metric is None:
        metric = list(results[0]['results'].keys())[0]

    # Display result per experiment
    for idx in indices:
        exp = results[idx]
        params_str = "" if params_off else f"Params: {exp['params']}"
        print(f"Index {idx}: {params_str} | {metric}: {exp['results'][metric]}")


def smart_formatter(val):
    """
    Formats a numerical value into a smart human-readable string:
    - Uses standard decimal format for values in [1e-2, 1e3)
    - Uses scientific notation for very small or large values
    - Formats with commas as decimal separators for better readability in some locales
    - Removes unnecessary trailing zeros and commas
    """
    if val == 0:
        return "0"
    
    abs_val = abs(val)
    
    # Use decimal format for moderate values
    if 1e-2 <= abs_val < 1e3:
        return f"{val:,.2f}".replace(",", " ").replace(".", ",").rstrip('0').rstrip(',')
    
    # Use scientific notation for very small or very large values
    exponent = int(np.floor(np.log10(abs_val)))
    base = round(val / (10 ** exponent), 1)
    
    if base == 1.0:
        return f"$10^{{{exponent}}}$"
    else:
        return f"${base}\\times10^{{{exponent}}}$"

def format_ticks_smart(axis, axis_type='x'):
    """
    Applies the smart formatter to the ticks of a given matplotlib axis.
    
    Parameters:
    - axis (matplotlib.axis): Axis object to format.
    - axis_type (str): Type of axis to format ('x' or 'y').
    """
    formatter = mticker.FuncFormatter(lambda val, _: smart_formatter(val))
    
    # Apply formatter to the specified axis
    if axis_type == 'x':
        axis.xaxis.set_major_formatter(formatter)
    else:
        axis.yaxis.set_major_formatter(formatter)


def assign_gradient_colors(sorted_keys, cmap_name='viridis'):
    """
    Assigns a distinct color from a colormap to each key in a sorted list.

    Parameters:
    - sorted_keys (list): List of keys to assign colors to (usually tuples of parameter values).
    - cmap_name (str): Name of the matplotlib colormap to use (default: 'viridis').

    Returns:
    - dict: A dictionary mapping each key to a color from the colormap.
    """
    cmap = get_cmap(cmap_name)  # Get the colormap object
    num_keys = len(sorted_keys)  # Total number of keys

    # Assign a color to each key by evenly spacing them along the colormap range [0,1]
    return {key: cmap(i / max(1, num_keys - 1)) for i, key in enumerate(sorted_keys)}

def plot_metrics_vs_param(results, param_x, metrics, group_by=None,
                          split_by=None, title="", grid=True, save_path=None,
                          ylim=None, log_scale_x=False, log_scale_y=False,
                          sub_plot=True, max_overall=False, show_plot=True, use_color_gradient=True,
                          font_scale=1.0, GT_plot=True, stds=None, dashed=False, fill_between=False,
                          line=False):
    """
    Plots one or several metrics against a given hyperparameter, grouping lines by other parameters,
    and splitting plots by yet another set of parameters.

    Parameters:
    - results (list): Output from parameter_scan containing all experiments.
    - param_x (str): Parameter name to be used on the X-axis.
    - metrics (list): List of metrics to be plotted on the Y-axis.
    - group_by (str or list): Parameter(s) used to distinguish curves within each plot.
    - split_by (str or list): Parameter(s) used to separate different panels (plots).
    - title (str): Global title of the plot or subplots.
    - grid (bool): Whether to show grid in each panel.
    - save_path (str): Path prefix to save the plots (PNG format).
    - ylim (tuple): Limits for the Y-axis.
    - log_scale_x / log_scale_y (bool): Whether to use log scale on X or Y axis.
    - sub_plot (bool): Whether to use subplots (True) or separate plots (False).
    - max_overall (bool): If True, plots only the best metric value for each x.
    - show_plot (bool): Whether to show the plot interactively.
    - use_color_gradient (bool): If True, use gradient coloring for grouped lines.
    - font_scale (float): Font scaling factor for labels and titles.
    - GT_plot (bool): Whether to include GT (ground truth) if available.
    - stds (dict or None): Dictionary of standard deviations per experiment (optional).
    - dashed (bool): If True, use dashed lines for the plot.
    - fill_between (bool): If True, use fill_between for uncertainty regions.
    - line (bool): If True, use plain lines (no markers).
    """
    
    # Normalize inputs: convert string to list if needed
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(split_by, str):
        split_by = [split_by]
    group_by = group_by or []
    split_by = split_by or []
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # Define plotting styles for each metric
    markers = ['o', 's', 'D', '^', 'v', 'x']
    linestyles = ['-', '--', '-.', ':']
    metric_styles = {
        metric: {
            'marker': markers[i % len(markers)],
            'linestyle': linestyles[i % len(linestyles)]
        } for i, metric in enumerate(metrics)
    }

    # Identify all combinations of values for split_by parameters
    unique_values = {key: sorted(set(exp['params'].get(key) for exp in results)) for key in split_by}
    all_combinations = list(product(*(unique_values[k] for k in split_by)))

    # Group experiments by split_key combination
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
        ncols = min(2, num_plots)  # Max 2 columns
        nrows = math.ceil(num_plots / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for lines

        # Iterate over each panel and plot
        for idx, (split_key, group_results) in enumerate(split_groups.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            _plot_one_panel(ax, group_results, param_x, metrics, group_by,
                            metric_styles, color_cycle, split_key, split_by,
                            title, grid, ylim, log_scale_x, log_scale_y, max_overall=max_overall, 
                            use_color_gradient=use_color_gradient, font_scale=font_scale, 
                            GT_plot=GT_plot, stds=stds, dashed=dashed, fill_between=fill_between, line=line)
            
            format_ticks_smart(ax, 'x')
            format_ticks_smart(ax, 'y')

        # Hide any unused subplot axes
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
                            use_color_gradient=use_color_gradient, max_overall=max_overall, font_scale=font_scale,
                            GT_plot=GT_plot, stds=stds, dashed=dashed, fill_between=fill_between, line=line)

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
                    max_overall=False, use_color_gradient=False, font_scale=1.0, 
                    GT_plot=True, stds=None, dashed=False, fill_between=False, line=False):
    """
    Internal helper function: plots a single panel (subplot or full figure).
    If max_overall=True, selects the best value across all other hyperparameters
    (except param_x, group_by, split_by).
    """

    # Group experiments by the values of `group_by` parameters
    grouped = defaultdict(list)
    for exp in group_results:
        group_key = tuple((k, exp['params'].get(k, None)) for k in group_by)
        grouped[group_key].append(exp)

    # Sort groups for consistent coloring
    sorted_keys = sorted(grouped.keys(), key=lambda x: [v for (_, v) in x])

    # Assign colors to each group
    color_map = assign_gradient_colors(sorted_keys) if use_color_gradient else {
        group: color_cycle[i % len(color_cycle)] for i, group in enumerate(sorted_keys)
    }

    # Loop over each group to compute and plot metrics
    for group_key in sorted_keys:
        exps = grouped[group_key]
        grouped_by_x = defaultdict(list)
        for exp in exps:
            x = exp['params'][param_x]
            grouped_by_x[x].append(exp)

        x_vals = sorted(grouped_by_x.keys())
        metric_vals = {metric: [] for metric in metrics}
        metric_errs = {metric: [] for metric in metrics}

        # Aggregate metric values per x value
        for x in x_vals:
            exp_list = grouped_by_x[x]
            for metric in metrics:
                best_mean = None
                best_err = None
                for exp in exp_list:
                    values = exp['results'][metric]
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], list):  # e.g., multiple repetitions
                            values = [v[-1] for v in values]  # Take last epoch value
                    elif isinstance(values, float):
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
                    # Select only best value across repetitions or configs
                    metric_vals[metric].append(best_mean)
                    metric_errs[metric].append(best_err)
                else:
                    # Average across repetitions for that (group, x)
                    all_means = []
                    all_errs = []
                    for exp in exp_list:
                        values = exp['results'][metric]
                        if isinstance(values, list) and len(values) > 0:
                            if isinstance(values[0], list):
                                values = [v[-1] for v in values]
                        elif isinstance(values, float):
                            values = [values]
                        all_means.append(np.mean(values))
                        all_errs.append(sem(values) if len(values) > 1 else 0.0)
                    metric_vals[metric].append(np.mean(all_means))
                    metric_errs[metric].append(np.mean(all_errs))

        # Plot the metric for the current group
        for metric in metrics:
            style = metric_styles[metric]
            # Build label depending on how many metrics are shown
            label_parts = [f"{format_display_name(k)}={format_display_name(v) if k == 'strategy' else v}" for k, v in group_key]
            label = (
                f"{format_display_name(metric)} ({', '.join(label_parts)})"
                if group_by and len(metrics) > 1
                else ", ".join(label_parts) if group_by
                else format_display_name(metric)
            )

            linestyle = "--" if dashed else (style['marker'] + style['linestyle'])
            yerrs = np.array(metric_errs[metric])

            if np.any(yerrs > 0) and not line:
                if fill_between:
                    ax.plot(x_vals, metric_vals[metric], linestyle, label=rf"{label}", color=color_map[group_key])
                    ax.fill_between(
                        x_vals,
                        np.array(metric_vals[metric]) - yerrs,
                        np.array(metric_vals[metric]) + yerrs,
                        color=color_map[group_key],
                        alpha=0.2
                    )
                else:
                    ax.errorbar(
                        x_vals,
                        metric_vals[metric],
                        yerr=yerrs,
                        fmt=linestyle,
                        capsize=5,
                        label=rf"{label}",
                        color=color_map[group_key]
                    )
            else:
                # If no error bars, simple line
                ax.plot(
                    x_vals,
                    metric_vals[metric],
                    linestyle,
                    label=rf"{label}",
                    color=color_map[group_key]
                )

    # Build the title including the split_key values
    split_label = ", ".join(f"{format_display_name(k)}={v}" for k, v in split_key) if split_by else ""
    full_title = f"{title}\\ {split_label}" if split_label else title
    ax.set_title(rf"{full_title}", fontsize=14 * font_scale)

    # Axis labels
    ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=12 * font_scale)
    ax.set_ylabel(
        rf"{', '.join(format_display_name(m) for m in metrics)}" if len(metrics) > 1 
        else rf"{format_display_name(metrics[0])}",
        fontsize=12 * font_scale
    )

    # Grid, scale, and limits
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    if ylim:
        ax.set_ylim(ylim)
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=11 * font_scale)

    # === Optionally add GT accuracy (for comparison) ===
    if (metrics == ["accuracy"]) and GT_plot:
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

    # Add legend
    ax.legend(fontsize=11 * font_scale)

def plot_optimal_param_vs_x(
    results,
    param_x,       # e.g., "s"
    parameter,     # e.g., "weight_decay"
    metric,        # e.g., "accuracy"
    group_by=None, # e.g., "K"
    log_scale_x=False,
    log_scale_y=False,
    save_path=None,
    font_scale=1.5,
    title=None,
    show_plot=True
):
    """
    For each value of `param_x`, this function identifies the optimal value of `parameter`
    that maximizes or minimizes the given `metric`, optionally grouped by a parameter.
    
    It then plots the optimal `parameter` value as a function of `param_x`.
    """

    # Determine whether to maximize or minimize the metric
    maximize = "loss" not in metric.lower() and "error" not in metric.lower()

    # Ensure group_by is a list
    group_by = [group_by] if isinstance(group_by, str) else (group_by or [])

    # Group experiments by (group_by, param_x)
    grouped_results = defaultdict(list)
    for exp in results:
        key = tuple((g, exp['params'][g]) for g in group_by)
        x_val = exp['params'][param_x]
        grouped_results[(key, x_val)].append(exp)

    # Dictionary to hold the final plotted curves
    curves = defaultdict(list)

    # For each group, find the best `parameter` value at each `param_x`
    for (group_key, x_val), exps in grouped_results.items():
        candidates = []
        for exp in exps:
            score = exp['results'][metric]
            score = np.mean(score) if isinstance(score, list) else score
            param_val = exp['params'][parameter]
            candidates.append((score, param_val))

        # Select the best parameter based on max or min score
        best = max(candidates, key=lambda x: x[0]) if maximize else min(candidates, key=lambda x: x[0])
        best_val = best[1]

        # Estimate error if multiple equivalent best scores exist
        matching_vals = [v for s, v in candidates if s == best[0]]
        err = sem(matching_vals) if len(matching_vals) > 1 else 0

        # Store for plotting
        curves[group_key].append((x_val, best_val, err))

    # Plotting section
    fig, ax = plt.subplots(figsize=(9, 6))
    for group_key, data in curves.items():
        data = sorted(data)
        x_vals = [x for x, y, _ in data]
        y_vals = [y for _, y, _ in data]
        y_errs = [e for _, _, e in data]
        label = ", ".join(f"{format_display_name(k)}={v}" for k, v in group_key) if group_by else None
        ax.errorbar(x_vals, y_vals, yerr=y_errs, label=label, capsize=4, marker='o')

    # Axis labels and title
    ax.set_xlabel(rf"{format_display_name(param_x)}", fontsize=12 * font_scale)
    ax.set_ylabel(rf"Optimal {format_display_name(parameter)}", fontsize=12 * font_scale)
    ax.set_title(
        title or rf"Optimal {format_display_name(parameter)} vs {format_display_name(param_x)} for {format_display_name(metric)}",
        fontsize=14 * font_scale
    )

    # Apply log scaling if required
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    # Add legend if grouped
    if group_by:
        ax.legend(fontsize=11 * font_scale)

    # Formatting
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis='both', labelsize=11 * font_scale)
    plt.tight_layout()

    # Save figure if path is specified
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"✅ Saved plot to {save_path}.png")

    # Show plot if enabled
    if show_plot:
        plt.show()



#######################################
# This function is never used in the codebase, but it is kept here for future use.
#######################################

def plot_histograms_from_results(results, metric, group_by=None, split_by=None, font_scale=1.0,
                                 error_type=None, title=None, save_path=None, bins_num=None, log_x=False,
                                 log_y=False):
    """
    Plot histograms or bar plots with error bars for a given metric from experiment results.
    Automatically flattens list-of-lists for metrics like 'slopes'.

    Parameters:
    - results: list of dicts with 'params' and 'results'
    - metric: str, the key in results['results']
    - group_by: str or list of str, parameter(s) to group by
    - split_by: str or list of str, parameter(s) to split subplots
    - font_scale: float, to scale fonts
    - error_type: None, 'std', or 'sem' (if set, uses bar plot with error bars instead of histogram)
    - title: optional global title (in LaTeX)
    - save_path: optional filename prefix to save
    """
    
    if not bins_num:
        bins_num = 'auto'
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(split_by, str):
        split_by = [split_by]
    group_by = group_by or []
    split_by = split_by or []

    split_dict = defaultdict(list)
    for exp in results:
        key = tuple((k, exp['params'][k]) for k in split_by) if split_by else [("All", "All")]
        split_dict[tuple(key)].append(exp)

    num_plots = len(split_dict)
    ncols = min(2, num_plots)
    nrows = -(-num_plots // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, (split_key, exps) in enumerate(split_dict.items()):
        ax = axes[idx]
        data = defaultdict(list)

        for exp in exps:
            values = exp["results"][metric]
            # Flatten if values is a list of lists
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = [v for sublist in values for v in sublist]
            elif not isinstance(values, list):
                values = [values]

            key = tuple(exp["params"].get(g, "All") for g in group_by) or ("All",)
            data[key].extend(values)

        if error_type in ["std", "sem"]:
            keys, means, errors = [], [], []
            for k, vals in sorted(data.items()):
                keys.append(k)
                means.append(np.mean(vals))
                errors.append(np.std(vals) if error_type == "std" else sem(vals))

            x_vals = np.arange(len(keys))
            formatted_labels = [", ".join(smart_formatter(float(val)) if isinstance(val, (int, float)) else str(val) for val in k) for k in keys]
            ax.bar(x_vals, means, yerr=errors, capsize=5, alpha=0.7)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(formatted_labels, rotation=30, ha='right', fontsize=10 * font_scale)
        else:
            for k, vals in data.items():
                label = ", ".join(map(str, k))
                ax.hist(vals, bins=bins_num, alpha=0.6, label=rf"{label}")

        if title:
            ax.set_title(rf"{title}", fontsize=14 * font_scale)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(rf"{format_display_name(metric)}", fontsize=12 * font_scale)
        metric_label = ", ".join([format_display_name(metric)]) if isinstance(metric, str) else ", ".join([format_display_name(m) for m in metric])
        ax.set_ylabel(rf"Number of {metric_label}", fontsize=12 * font_scale)
        ax.legend(fontsize=10 * font_scale)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10 * font_scale)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    plt.show()
