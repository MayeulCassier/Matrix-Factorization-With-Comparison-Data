# all imports

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd


############################################
# These functions are used to visualize the results, parameters or to get the best parameters
############################################

# Plotting the losses
def plot_losses(results, param_index=None):
    """
    Plots training and validation losses.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_index (int, optional): The index of a specific experiment to plot train & val losses on the same graph.
      If None, plots all training losses in one graph and all validation losses in another.
    """
    if param_index is not None:
        # Plot a specific experiment's train and validation loss on the same graph
        exp = results[param_index]
        train_losses = exp['results']['train_losses']
        val_losses = exp['results']['val_losses']
        
        plt.figure(figsize=(10, 5))
        for rep in range(len(train_losses)):
            plt.plot(train_losses[rep], label=f'Train Loss (rep {rep+1})', linestyle='--')
            plt.plot(val_losses[rep], label=f'Val Loss (rep {rep+1})')
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train & Val Loss for {exp['params']}")
        plt.legend()
        plt.show()
    else:
        # Plot all train losses in one graph
        plt.figure(figsize=(10, 5))
        for i, exp in enumerate(results):
            for rep in range(len(exp['results']['train_losses'])):
                plt.plot(exp['results']['train_losses'][rep], label=f'Exp {i+1}, Rep {rep+1}')
        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title("Training Losses for All Experiments")
        plt.legend()
        plt.show()

        # Plot all validation losses in one graph
        plt.figure(figsize=(10, 5))
        for i, exp in enumerate(results):
            for rep in range(len(exp['results']['val_losses'])):
                plt.plot(exp['results']['val_losses'][rep], label=f'Exp {i+1}, Rep {rep+1}')
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title("Validation Losses for All Experiments")
        plt.legend()
        plt.show()

# heatmap between two parameters for maximum value of a result metric
def plot_heatmap_best_fixed(results, param_x, param_y, result_metric):
    """
    Finds the best configuration for a given result metric and generates a heatmap
    by fixing all parameters except param_x and param_y.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    """
    
    # Step 1: Find the best experiment based on the highest metric value
    best_exp = max(results, key=lambda exp: max(exp['results'][result_metric]))
    best_params = best_exp['params']
    
    print(f"Best configuration found: {best_params}, {result_metric}: {max(best_exp['results'][result_metric])}")
    
    # Step 2: Fix all parameters except param_x and param_y
    data = {}
    
    for exp in results:
        # Check if all other parameters match the best configuration
        if all(exp['params'][key] == best_params[key] for key in best_params if key not in [param_x, param_y]):
            x_val = exp['params'][param_x]
            y_val = exp['params'][param_y]
            metric_val = max(exp['results'][result_metric])  # Best result for this configuration
            
            if (x_val, y_val) not in data:
                data[(x_val, y_val)] = metric_val
            else:
                data[(x_val, y_val)] = max(data[(x_val, y_val)], metric_val)
    
    # Step 3: Prepare data for heatmap
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    
    for (x, y), value in data.items():
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_matrix[y_idx, x_idx] = value
    
    # Step 4: Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(heatmap_matrix, xticklabels=x_values, yticklabels=y_values, annot=True, cmap="coolwarm")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(f"Heatmap of {result_metric} by {param_x} and {param_y}\n(Fixed best params: {best_params})")
    plt.show()


# heatmap between two parameters for maximum value of a result metric with fixed values for other parameters
def plot_heatmap_fixed(results, param_x, param_y, result_metric, fixed_index, ax = None):
    """
    Plots a heatmap of two chosen parameters against a selected result metric, with fixed values for other parameters.
    
    Parameters:
    - results (list): The output from parameter_scan.
    - param_x (str): Name of the first parameter.
    - param_y (str): Name of the second parameter.
    - result_metric (str): The result metric to use for coloring the heatmap.
    - fixed_index (int): The index of the experiment that should be used to fix other parameters.

    """
    fixed_params = results[fixed_index]['params']
    data = {}
    

    
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
    sns.heatmap(heatmap_matrix, xticklabels=x_values, yticklabels=y_values, annot=True, cmap="coolwarm", ax=ax)

    # Set labels and title
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Heatmap of {result_metric} by {param_x} and {param_y} (Fixed at Index {fixed_index})")

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
    - result_metric (str): The result metric to maximize.
    
    Returns:
    - dict: The best hyperparameter configuration.
    """
    best_exp = max(results, key=lambda exp: max(exp['results'][result_metric]))
    best_params = best_exp['params']
    best_value = max(best_exp['results'][result_metric])
    
    print(f"Best parameters for {result_metric}: {best_params}, Best value: {best_value}")
    return best_params

# Gets best hyperparameter configuration for all result metrics
def get_best_params_all_metrics(results):
    """
    Finds the best hyperparameter configuration for each result metric in the dataset.
    
    Parameters:
    - results (list): The output from parameter_scan.
    
    Returns:
    - dict: A dictionary where keys are result metrics and values are the best parameter configurations.
    """
    all_metrics = results[0]['results'].keys()  # Get all result metrics from the first experiment
    best_params_per_metric = {}
    
    for metric in all_metrics:
        best_params_per_metric[metric] = get_best_params(results, metric)
    
    return best_params_per_metric