import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


class Model:
    def __init__(self):
        self.created_timestamp = str(datetime.date.today())

    def add_data(self, df: pd.DataFrame, target_col: "str" = ""):

        if len(target_col) > 0:
            self.target = df[target_col]

        self.features = df.drop(target_col, axis=1)

    def add_model(self, model):
        self.model = model


def validate_model_cross(
    model,
    feature_matrix: pd.DataFrame,
    target: pd.DataFrame,
    random_state: int = 1,
    n_splits: int = 5,
) -> dict:
    """Function that runs stratified cross validation for a model
    on a train_predict dataset.

    model: sklearn (or similar) model object ready to be fit
    feature_matrix: pd.DataFrame holding features that will be used for
        model train_predict/cross validation
    target: pd.DataFrame holding targets that will be used for
        model train_predict/cross validation
    random_state: Random seed to split data by
    n_splits: Number of folds to use for cross-validation

    _______
    Returns: dict with the true target value and prediction probabilities. Each
        prediction comes from the out of fold predictions during cross validation.
    """

    y_pred_probas = cross_val_predict(
        model,
        feature_matrix,
        np.reshape(target.values, (target.shape[0],)),
        cv=StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True),
        n_jobs=-1,
        method="predict_proba",
    )

    target_predictions = {
        "targets": target.values,
        "predictions": y_pred_probas[:, 1],
    }

    return target_predictions


def get_performance_metrics(target_predictions_dict, classifier_threshold):
    """Given the true labels and predictions, calculate a few key model performance
    metrics

    target_predictions_dict: dict with keys 'targets' and 'predictions'. Each item is a
        1-d array (or list)
    classifier_threshold: float representing the threshold to use on the prediction
        probabilities to decide which class the prediction belongs to.

    _______
    Returns: dict with a range of different performance metrics
    """

    labels = target_predictions_dict["targets"]
    pred_proba = target_predictions_dict["predictions"]
    pred_labels = pred_proba >= classifier_threshold

    return {
        "baseline_accuracy": np.round(labels.mean(), 2),
        "accuracy": accuracy_score(labels, pred_labels),
        "precision": precision_score(labels, pred_labels),
        "recall": recall_score(labels, pred_labels),
        "AUROC": roc_auc_score(labels, pred_proba),
    }
