import logging
from json import dump as json_dump
from json import load as json_load
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists

from numpy import array as np_array
from numpy import dot as np_dot
from numpy.linalg import norm as np_norm

from scripts.compute_robustness_thresholds import compute_thresholds
from scripts.extract_class_text_features import extract_clip

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


def val_robustness_check():
    model_name = "clip-vit-large-patch14/"
    val_img_features_folder = f"data/features/{model_name}/val/json"
    if not path_exists(val_img_features_folder):
        msg = f"Folder '{val_img_features_folder}' does not exist."
        raise ValueError(msg)

    modified_class_mapping_path = "data/modified_class_mapping.json"
    if not path_exists(modified_class_mapping_path):
        raise ValueError("data/modified_class_mapping.json does not exist.")

    train_text_features_json_path = "data/train_class_text_features.json"
    if not path_exists(train_text_features_json_path):
        extract_clip(add_stream_handler=False)

    train_thresholds_path = "data/train_robustness_thresholds.json"
    if not path_exists(train_thresholds_path):
        compute_thresholds(add_stream_handler=False)

    with open(modified_class_mapping_path, "r") as f:
        modified_class_mapping = json_load(f)

    with open(train_text_features_json_path, "r") as f:
        train_text_features_json = json_load(f)

    with open(train_thresholds_path, "r") as f:
        train_thresholds = json_load(f)

    robustness_folder = "data/robustness_check/val"
    if not path_exists(robustness_folder):
        os_makedirs(robustness_folder)

    robustness_check_results = {}
    for json_filename_1 in os_listdir(val_img_features_folder):
        robustness_data = {}
        total_in_class = 0
        total_outside_class = 0
        true_positive_num = 0
        true_negative_num = 0

        with open(f"{val_img_features_folder}/{json_filename_1}", "r") as f:
            val_img_features_dict = json_load(f)

        val_id = json_filename_1.replace("_features.json", "")
        val_class_name = modified_class_mapping[val_id][0]
        val_text_features = train_text_features_json[val_class_name]
        val_text_features = np_array(val_text_features)

        robustness_data["class_name"] = val_class_name
        robustness_data["in_class"] = {}
        robustness_data["outside_class"] = {}

        for img_filename in val_img_features_dict:
            total_in_class += 1
            val_img_features = val_img_features_dict[img_filename]
            val_img_features = np_array(val_img_features)

            # Compute cosine similarity
            cos_sim = np_dot(val_img_features, val_text_features) / (
                np_norm(val_img_features) * np_norm(val_text_features)
            )

            if cos_sim > train_thresholds[val_class_name]:
                true_positive_num += 1

            robustness_data["in_class"][img_filename] = cos_sim

        for json_filename_2 in os_listdir(val_img_features_folder):
            if json_filename_1 != json_filename_2:
                feat_filepath = f"{val_img_features_folder}/{json_filename_2}"
                with open(feat_filepath, "r") as f:
                    val_img_features_dict = json_load(f)

                curr_id = json_filename_2.replace("_features.json", "")
                curr_class_name = modified_class_mapping[curr_id][0]

                for img_filename in val_img_features_dict:
                    total_outside_class += 1
                    val_img_features = val_img_features_dict[img_filename]
                    val_img_features = np_array(val_img_features)

                    # Compute cosine similarity
                    cos_sim = np_dot(val_img_features, val_text_features) / (
                        np_norm(val_img_features) * np_norm(val_text_features)
                    )

                    if cos_sim < train_thresholds[val_class_name]:
                        true_negative_num += 1

                    curr_key_name = f"{img_filename} [{curr_class_name}]"
                    robustness_data["outside_class"][curr_key_name] = cos_sim

        for key_name in ["in_class", "outside_class"]:
            robustness_data[key_name] = {
                item[0]: item[1]
                for item in sorted(
                    robustness_data[key_name].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

        with open(f"{robustness_folder}/{val_id}.json", "w") as f:
            json_dump(robustness_data, f, indent=4)

        true_positive_rate = true_positive_num / total_in_class
        true_negative_rate = true_negative_num / total_outside_class

        individual_results = {
            "total_in_class": total_in_class,
            "total_outside_class": total_outside_class,
            "true_positive_num": true_positive_num,
            "true_negative_num": true_negative_num,
            "true_positive_rate": true_positive_rate,
            "true_negative_rate": true_negative_rate,
        }
        robustness_check_results[val_class_name] = individual_results
        logger.info(f"{val_class_name}: {individual_results}")

    with open("data/val_robustness_check_results.json", "w") as f:
        json_dump(robustness_check_results, f, indent=4)


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.val_robustness_check
    val_robustness_check()
