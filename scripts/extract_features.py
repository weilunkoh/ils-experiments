import argparse
import logging
from json import dump as json_dump
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists

from PIL.Image import open as Image_open
from torch import load as torch_load
from torch import no_grad
from torch import save as torch_save
from torch.cuda import is_available as cuda_is_available

from src.helper.common_values import EMBED_MODEL_NAMES
from src.helper.load_model import load_model_and_processor

logger = logging.getLogger()


def extract_features(
    embed_model_name: str,
    data_folder_name: str,
    embed_model_version: str = None,
    abs_path_prefix: str = "",
    output_already_logs: bool = True,
    add_stream_handler: bool = True,
) -> str:
    if add_stream_handler:
        logger.addHandler(logging.StreamHandler())

    # Check if configuration values are allowed
    if embed_model_name not in EMBED_MODEL_NAMES:
        raise ValueError(f"Model name must be one of {EMBED_MODEL_NAMES}.")

    # Check if source folder exists
    source_folder = f"{abs_path_prefix}/data/{data_folder_name}"
    if not path_exists(source_folder):
        raise ValueError(f"Data folder '{data_folder_name}' does not exist.")

    # Load model and processor
    model, processor, embed_model_version = load_model_and_processor(
        embed_model_name, embed_model_version=embed_model_version
    )

    # Make target folder for features
    target_folder_1 = f"{abs_path_prefix}/data/features"
    target_folder_2 = f"{embed_model_version}/{data_folder_name}"
    target_folder = f"{target_folder_1}/{target_folder_2}"
    os_makedirs(target_folder, exist_ok=True)

    # Check for GPU
    if cuda_is_available():
        device = "cuda"
        logger.info("GPU available. Loading tensors to GPU.")
    else:
        device = "cpu"
        logger.info("GPU not available. Loading tensors to CPU.")

    # Extract features for each image
    for class_folder in os_listdir(source_folder):
        image_names = os_listdir(f"{source_folder}/{class_folder}")
        num_images = len(image_names)
        target_class_folder = f"{target_folder}/{class_folder}"

        if not path_exists(target_class_folder):
            common_msg = (
                f"features for {num_images} images in class {class_folder}..."
            )
            logger.info(f"Extracting {common_msg}")
            os_makedirs(target_class_folder, exist_ok=True)

            for image in image_names:
                image_name = f"{source_folder}/{class_folder}/{image}"
                target_name = f"{target_folder}/{class_folder}/{image}.pt"

                with Image_open(image_name) as image:
                    inputs = processor(images=image, return_tensors="pt")
                    inputs.pixel_values = inputs.pixel_values.to(device)
                    with no_grad():
                        if embed_model_name == "inceptionV3":
                            outputs = model.forward_features(
                                inputs.pixel_values
                            )
                            pooled_outputs = model.global_pool(outputs)
                            features = pooled_outputs[0].squeeze()
                        else:
                            outputs = model(inputs.pixel_values)
                            if embed_model_name == "clip":
                                features = outputs.image_embeds[0]
                            else:
                                features = outputs[0].squeeze()
                        torch_save(features, target_name)

            logger.info(f"Extracted {common_msg}")
        else:
            if output_already_logs:
                msg = f"features for class {class_folder} already exist."
                logger.info(msg)

        # Save features in json
        os_makedirs(f"{target_folder}/json", exist_ok=True)
        feature_name = f"{target_folder}/json/{class_folder}_features.json"
        if not path_exists(feature_name):
            logger.info(f"Saving features for class {class_folder} in json...")
            features = {}
            for image in image_names:
                target_name = f"{target_folder}/{class_folder}/{image}.pt"
                with no_grad():
                    image_tensor = torch_load(target_name)
                    features[image] = image_tensor.tolist()
            with open(feature_name, "w") as f:
                json_dump(features, f)
            logger.info(f"Saved features in json for class {class_folder}.")
        else:
            if output_already_logs:
                logger.info(
                    f"Features for class {class_folder} already saved in json."
                )

    return embed_model_version


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.extract_features
    # --embed_model_name <embed_model_name> --data_folder_name <data_folder_name>
    # e.g. python -m scripts.extract_features --embed_model_name clip --data_folder_name val
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str)
    parser.add_argument("--data_folder_name", type=str)
    args = parser.parse_args()

    embed_model_name = args.embed_model_name
    data_folder_name = args.data_folder_name
    extract_features(embed_model_name, data_folder_name)
