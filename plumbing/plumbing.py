import argparse
import json
import os
from custom_data_pipeline import run_data_pipeline
from custom_train_predict_pipeline import run_train_pipeline
from constants import ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--pipeline",
    type=str,
    choices=["data_pipeline", "train", "predict"],
    required=True,
    help="Pipeline (stages) to run from the application",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="pipeline_test.json",
    help="Name you want files written to file to have.",
)


def run_msg(args, config):
    print(f"You are running pipeline {args.pipeline} with config:")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.pipeline == "data_pipeline":
        config_file = os.path.join(
            ROOT_DIR, "pipeline_configs", "data_processing", args.config
        )
        with open(config_file, "r") as f:
            config = json.load(f)

        run_msg(args, config)
        run_data_pipeline(**config)

    elif args.pipeline == "train":
        config_file = os.path.join(
            ROOT_DIR, "pipeline_configs", "train_predict", args.config
        )
        with open(config_file, "r") as f:
            config = json.load(f)

        run_msg(args, config)
        run_train_pipeline(mode="train", **config)

    elif args.pipeline == "predict":
        config_file = os.path.join(
            ROOT_DIR, "pipeline_configs", "train_predict", args.config
        )
        with open(config_file, "r") as f:
            config = json.load(f)

        run_msg(args, config)
        run_train_pipeline(mode="predict", **config)
