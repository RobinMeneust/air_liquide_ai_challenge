import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_prices(dfs: List, labels=None, title='Electricity Price Evolution', xlabel='Time (day)', ylabel='Price'):
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
    n_before = len(y_true) - len(predictions)
    
    if data_normalizer:
        y_true = data_normalizer.inverse_transform_numpy(y_true.reshape(-1, 1), "price")
        predictions = data_normalizer.inverse_transform_numpy(predictions.reshape(-1, 1), "price")
    
    x_pred = x_dates
    
    if n_before >= 1:
        x_pred = x_pred[n_before-1:]
    
        # We add at beginning of predictions the last ground_truth values to have a continuous plot
        predictions = np.insert(predictions, 0, y_true[n_before-1], axis=0)
        
    plt.figure(figsize=(12,5))
    plt.plot(x_dates, y_true, color = 'r', label="True", marker='o')
    plt.plot(x_pred, predictions, color = 'b', label="Prediction", marker='x', linestyle='--')
    plt.title('Real and Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d\n%H h"))
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()