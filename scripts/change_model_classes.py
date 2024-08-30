import argparse
import logging
from json import load as json_load
from os.path import exists as path_exists

from joblib import dump as joblib_dump
from joblib import load as joblib_load
from numpy import array as np_array

logger = logging.getLogger()


def change_model_classes(artifacts_timestamp: str):
    logger.addHandler(logging.StreamHandler())

    modified_class_mapping_path = "data/modified_class_mapping.json"
    if not path_exists(modified_class_mapping_path):
        raise ValueError(f"{modified_class_mapping_path} does not exist.")

    # Load model
    folder_path = f"outputs/{artifacts_timestamp}/artifacts"
    model_path = f"{folder_path}/model.joblib"
    model = joblib_load(model_path)

    # Load class_mapping
    with open("data/modified_class_mapping.json", "r") as f:
        modified_class_mapping = json_load(f)

    # Get new classes
    new_classes = [modified_class_mapping[c][0] for c in model.classes_]

    # Change model classes
    model.classes_ = np_array(new_classes)

    # Save new model
    new_model_path = f"{folder_path}/model_with_class_names.joblib"
    joblib_dump(model, new_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_timestamp", type=str)
    args = parser.parse_args()

    artifacts_timestamp = args.artifacts_timestamp
    change_model_classes(artifacts_timestamp)
