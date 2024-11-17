import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(y):
    """
    Plot the distribution of classes in the dataset.

    Parameters:
    - y: array-like, the class labels of the dataset
    """
    class_counts = pd.Series(y).value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title("Class Distribution in Dataset")
    plt.ylabel('')
    plt.show()


def plot_cost_distribution(cost_data):
    """
    Plot the cost distribution as a pie chart.

    Parameters:
    - cost_data: dict, with keys as cost categories and values as costs.
    """
    labels = list(cost_data.keys())
    sizes = list(cost_data.values())

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Cost Distribution")
    plt.show()
