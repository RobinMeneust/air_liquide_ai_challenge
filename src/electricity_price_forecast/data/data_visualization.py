import matplotlib.pyplot as plt
from typing import List

def plot_prices(dfs: List, labels=None, title='Electricity Price Evolution', xlabel='Time (day)', ylabel='Price'):
    if labels is not None and len(dfs) != len(labels):
        raise ValueError("Number of labels must match number of dataframes if provided")
    
    plt.figure(figsize=(15,5))
    for i, df in enumerate(dfs):
        if labels:
            plt.plot(df.index, df["price"], label=labels[i])
        else:
            plt.plot(df.index, df["price"])
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()