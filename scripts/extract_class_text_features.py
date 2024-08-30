import logging
from json import dump as json_dump
from json import load as json_load
from os.path import exists as path_exists

from torch import no_grad
from torch.cuda import is_available as cuda_is_available

from src.helper.clip_robustness_check import load_clip_text

logger = logging.getLogger()


def extract_clip(
    add_stream_handler: bool = True,
) -> str:
    if add_stream_handler:
        logger.addHandler(logging.StreamHandler())

    modified_class_mapping_path = "data/modified_class_mapping.json"

    if not path_exists(modified_class_mapping_path):
        raise ValueError("data/modified_class_mapping.json does not exist.")
    with open(modified_class_mapping_path, "r") as f:
        modified_class_mapping = json_load(f)

    train_class_text = [v[0] for v in modified_class_mapping.values()]

    text_model, tokenizer = load_clip_text()

    # Check for GPU
    if cuda_is_available():
        device = "cuda"
        logger.info("GPU available. Loading tensors to GPU.")
    else:
        device = "cpu"
        logger.info("GPU not available. Loading tensors to CPU.")

    text_inputs = tokenizer(
        train_class_text, return_tensors="pt", padding=True
    )
    text_inputs.input_ids = text_inputs.input_ids.to(device)
    text_inputs.attention_mask = text_inputs.attention_mask.to(device)
    with no_grad():
        text_outputs = text_model(
            text_inputs.input_ids, text_inputs.attention_mask
        )
        text_features = text_outputs.text_embeds

    text_features_list = text_features.tolist()

    train_features_json = {
        k: v for k, v in zip(train_class_text, text_features_list)
    }

    train_features_json_path = "data/train_class_text_features.json"
    with open(train_features_json_path, "w") as f:
        json_dump(train_features_json, f, indent=4)


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.extract_class_text_features
    extract_clip()
