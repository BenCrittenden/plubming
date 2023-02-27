import json
import os
import pandas as pd
from constants import ROOT_DIR
from train_model import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def load_matrix_data(matrix_filename) -> pd.DataFrame:

    return pd.read_feather(
        os.path.join(
            ROOT_DIR, "data", "transformed", "application_credit_matrix.feather"
        )
    ).set_index("ID")


def run_train_pipeline(
    gold_filename,
    hyperparameters,
    classifier_threshold,
    model_filename,
):

    df_features_target = load_matrix_data(gold_filename)

    model_obj = Model()
    model_obj.add_data(df_features_target, "APPROVED")
    model_obj.add_model(GradientBoostingClassifier(**hyperparameters))

    X_train, X_test, y_train, y_test = train_test_split(
        model_obj.features,
        model_obj.target,
        test_size=0.2,
        random_state=1,
        stratify=model_obj.target,
    )

    metrics = get_performance_metrics(cv_target_predictions_dict, classifier_threshold)

    model_training_report = dict(
        type="cross validation report",
        date=model_obj.created_timestamp,
        trained_on=gold_filename,
        model=type(model_obj.model).__name__,
        hyperparameters=hyperparameters,
        classifier_threshold=classifier_threshold,
        metrics=metrics,
    )

    print(json.dumps(model_training_report, indent=2))

    report_file = os.path.join(
        ROOT_DIR, "models", "reports", f"{model_filename}_cv.json"
    )
    with open(report_file, "w") as f:
        json.dump(model_training_report, f, indent=2)
