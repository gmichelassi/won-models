import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import plot_confusion_matrix


def get_dataset_correlation(x: pd.DataFrame):
    plt.figure(figsize=(15, 15), tight_layout=True)
    corr = x.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        annot=True
    )
    plt.savefig('correlation.png')


def get_trees(model: RandomForestClassifier, feature_names: np.ndarray, class_names: np.ndarray):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=800)
    for index, current_tree in enumerate(model.estimators_):
        plot_tree(
            current_tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True
        )
        fig.savefig(f'rf_{index}.png')


def get_confusion_matrix(model: RandomForestClassifier, x_test: np.ndarray, y_test: np.ndarray):
    color = 'white'
    matrix = plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues)
    matrix.ax_.set_title('Confusion Matrix', color=color)
    plt.xlabel('Predicted Label', color=color)
    plt.ylabel('True Label', color=color)
    plt.gcf().axes[0].tick_params(colors=color)
    plt.gcf().axes[1].tick_params(colors=color)
    plt.savefig('confusion_matrix.png')
