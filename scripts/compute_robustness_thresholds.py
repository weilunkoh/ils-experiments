import logging
from json import dump as json_dump
from json import load as json_load
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists

from numpy import array as np_array
from numpy import dot as np_dot
from numpy.linalg import norm as np_norm

from scripts.extract_class_text_features import extract_clip

logger = logging.getLogger()


def compute_thresholds(add_stream_handler: bool = True):
    if add_stream_handler:
        logger.addHandler(logging.StreamHandler())

    model_name = "clip-vit-large-patch14/"
    train_img_features_folder = f"data/features/{model_name}/train/json"
    if not path_exists(train_img_features_folder):
        msg = f"Folder '{train_img_features_folder}' does not exist."
        raise ValueError(msg)

    modified_class_mapping_path = "data/modified_class_mapping.json"
    if not path_exists(modified_class_mapping_path):
        raise ValueError(f"{modified_class_mapping_path} does not exist.")

    train_text_features_json_path = "data/train_class_text_features.json"
    if not path_exists(train_text_features_json_path):
        extract_clip(add_stream_handler=False)

    with open(modified_class_mapping_path, "r") as f:
        modified_class_mapping = json_load(f)

    with open(train_text_features_json_path, "r") as f:
        train_text_features_json = json_load(f)

    robustness_folder = "data/robustness_check/train"
    if not path_exists(robustness_folder):
        os_makedirs(robustness_folder)

    thresholds = {}
    for json_filename in os_listdir(train_img_features_folder):
        robustness_data = {}
        with open(f"{train_img_features_folder}/{json_filename}", "r") as f:
            train_img_features_dict = json_load(f)

        train_id = json_filename.replace("_features.json", "")
        train_class_name = modified_class_mapping[train_id][0]
        train_text_features = train_text_features_json[train_class_name]
        train_text_features = np_array(train_text_features)

        robustness_data["class_name"] = train_class_name
        robustness_data["cos_sim"] = {}
        lowest_cos_sim = 1
        for img_filename in train_img_features_dict:
            train_img_features = train_img_features_dict[img_filename]
            train_img_features = np_array(train_img_features)

            # Compute cosine similarity
            cos_sim = np_dot(train_img_features, train_text_features) / (
                np_norm(train_img_features) * np_norm(train_text_features)
            )

            if cos_sim < lowest_cos_sim:
                lowest_cos_sim = cos_sim
                msg = f"{train_class_name}: {lowest_cos_sim} ({img_filename})"
                logger.info(msg)

            robustness_data["cos_sim"][img_filename] = cos_sim

        robustness_data["cos_sim"] = {
            item[0]: item[1]
            for item in sorted(
                robustness_data["cos_sim"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        with open(f"{robustness_folder}/{train_id}.json", "w") as f:
            json_dump(robustness_data, f, indent=4)
        thresholds[train_class_name] = lowest_cos_sim

    with open("data/train_robustness_thresholds.json", "w") as f:
        json_dump(thresholds, f, indent=4)


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.compute_robustness_thresholds
    compute_thresholds()
