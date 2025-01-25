import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_prices(dfs: List, labels=None, title='Electricity Price Evolution', xlabel='Time (day)', ylabel='Price'):
    """Plot the prices of the dataframes
    
    Args:
        dfs (List): List of dataframes to plot
        labels (List): List of labels for each dataframe (either None or same length as dfs)
        title (str): Title of the plot
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
    """
    if labels is not None and len(dfs) != len(labels):
        raise ValueError("Number of labels must match number of dataframes if provided")
    
    plt.figure(figsize=(15,5))
    for i, df in enumerate(dfs):
        if labels:
            plt.plot(df["date"], df["price"], label=labels[i])
        else:
            plt.plot(df["date"], df["price"])
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()
    
    
def plot_predictions_vs_real(x_dates, y_true, predictions, data_normalizer=None, save_path=False):
    """Plot the real and predicted prices. This fucntion expects len(y_true) > len(predictions)
    
    Args:
        x_dates (np.ndarray): Dates that appear on the plot (for history and predictions)
        y_true (np.ndarray): True prices (for history and predictions)
        predictions (np.ndarray): Predicted prices)
        data_normalizer (DataNormalizer): DataNormalizer object to inverse transform the data
        save_path (str): Path to save the plot (if None, the plot is displayed)
    """
    n_before = len(y_true) - len(predictions)
    if n_before <= 0:
        raise ValueError("Number of true values must be greater than number of predictions")
    
    if data_normalizer:
        y_true = data_normalizer.inverse_transform_numpy(y_true.reshape(-1, 1), "price")
        predictions = data_normalizer.inverse_transform_numpy(predictions.reshape(-1, 1), "price")
    

    x_before = x_dates[:n_before]
    x_after = x_dates[n_before-1:]
    
    y_before = y_true[:n_before]
    y_true = y_true[n_before-1:]

    # We add at beginning of predictions the last ground_truth values to have a continuous plot
    predictions = np.insert(predictions, 0, y_true[0], axis=0)
        
    plt.figure(figsize=(12,5))
    plt.plot(x_before, y_before, color = 'black', label="History")
    plt.plot(x_after, y_true, color = 'g', label="True")
    plt.plot(x_after, predictions, color = 'r', label="Prediction", linestyle='--')
    plt.title('Real and Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d\n%H h"))
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()