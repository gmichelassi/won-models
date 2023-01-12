import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data, preprocess_data
from plots import get_trees, get_confusion_matrix, get_dataset_correlation
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pprint import pp
import shap


RANDOM_STATE = 10


def fit_and_predict(
    x: np.ndarray,
    y: np.ndarray,
    model: RandomForestClassifier,
    get_probabilitities: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=RANDOM_STATE)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    if get_probabilitities:
        y_probablities = model.predict_proba(x_test)
        first_class, second_class = model.classes_

        probabilites = {
            first_class: [],
            second_class: []
        }

        for class_a_prob, class_b_prob in y_probablities:
            probabilites[first_class].append(class_a_prob)
            probabilites[second_class].append(class_b_prob)

        pp(probabilites)

    return x_train, x_test, y_train, y_test


def get_feature_importances(feature_names: pd.Index, model: RandomForestClassifier):
    importance_and_feature = {feature: importance for feature, importance in
                              zip(feature_names, model.feature_importances_)}
    sorted_importance_and_feature = sorted(importance_and_feature.items(), key=lambda v: v[1])

    pp(sorted_importance_and_feature)


def run_model():
    x, y = load_data()
    feature_names = x.columns
    x, y = preprocess_data(x, y)

    x = pd.DataFrame(x, columns=feature_names)

    model = RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    x_train, x_test, _, y_test = fit_and_predict(x, y, model, get_probabilitities=False)
    get_feature_importances(feature_names, model)

    class_names = model.classes_

    # get_trees(model, feature_names.to_numpy(), class_names)
    # get_confusion_matrix(model, x_test, y_test)

    explainer = shap.Explainer(model.predict, pd.DataFrame(x_train, columns=feature_names))
    shap_values = explainer(x_train)

    plt.figure(figsize=(15, 15), tight_layout=True)
    shap.plots.beeswarm(shap_values)
    plt.savefig('shap.png')


if __name__ == '__main__':
    run_model()
