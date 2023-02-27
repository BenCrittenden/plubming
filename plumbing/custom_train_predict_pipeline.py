import json
import os
import pickle
from constants import ROOT_DIR
from train_model import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def load_matrix_data(matrix_filename) -> pd.DataFrame:

    return pd.read_feather(
        os.path.join(ROOT_DIR, "data", "transformed", matrix_filename)
    ).set_index("ID")


def run_train_pipeline(
    mode,
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

    if mode == "train":
        # Now run cross validation on the train set
        target_predictions_dict = validate_model_cross(
            model_obj.model, X_train, y_train, random_state=1, n_splits=5
        )
        report_type = "cross validation report"

    elif mode == "predict":
        model_obj.model.fit(X_train, y_train)
        predictions = model_obj.model.predict_proba(X_test)
        target_predictions_dict = dict(targets=y_test, predictions=predictions[:, 1])
        report_type = "prediction report"

    metrics = get_performance_metrics(target_predictions_dict, classifier_threshold)

    model_report = dict(
        type=report_type,
        date=model_obj.created_timestamp,
        trained_on=gold_filename,
        model=type(model_obj.model).__name__,
        hyperparameters=hyperparameters,
        classifier_threshold=classifier_threshold,
        metrics=metrics,
    )

    print(json.dumps(model_report, indent=2))

    if mode == "train":
        report_file = os.path.join(
            ROOT_DIR, "models", "reports", f"{model_filename}_train.json"
        )
        with open(report_file, "w") as f:
            json.dump(model_report, f, indent=2)

    elif mode == "predict":

        model_report["model_filename"] = f"{model_filename}.pkl"
        report_file = os.path.join(
            ROOT_DIR, "models", "reports", f"{model_filename}_predict.json"
        )
        with open(report_file, "w") as f:
            json.dump(model_report, f, indent=2)

        model_file = os.path.join(ROOT_DIR, "models", f"{model_filename}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model_obj, f)
