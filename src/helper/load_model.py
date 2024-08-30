import argparse
import logging
from typing import Tuple

from timm import create_model as timm_create_model  # for inceptionV3
from torch.cuda import is_available as cuda_is_available
from torch.nn import Module as NNModule
from torch.nn import Sequential as nn_sequential
from torchvision.models import ResNet50_Weights, resnet50
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.utils import constants

from src.helper.common_values import EMBED_MODEL_NAMES

# from transformers.image_processing_utils import BaseImageProcessor
# Can consider replacing image mean and image std of CLIPImageProcessor
# and use it for other models pre-trained on imagenet data

logger = logging.getLogger()


def load_model_and_processor(
    embed_model_name: str,
    embed_model_version: str = None,
    set_eval: bool = True,
) -> Tuple[NNModule, CLIPImageProcessor]:
    if embed_model_name not in EMBED_MODEL_NAMES:
        raise ValueError(f"Model name must be one of {EMBED_MODEL_NAMES}.")

    if embed_model_name == "clip":
        if embed_model_version is None:
            embed_model_version = "clip-vit-large-patch14"
        clip_embed_model_name = f"openai/{embed_model_version}"
        logger.info(f"Loading CLIP model {clip_embed_model_name}...")
        model = CLIPVisionModelWithProjection.from_pretrained(
            clip_embed_model_name
        )
        processor = CLIPImageProcessor.from_pretrained(clip_embed_model_name)
    elif embed_model_name == "inceptionV3":
        processor = CLIPImageProcessor(
            size={"shortest_edge": 299},
            crop_size={"height": 299, "width": 299},
            image_mean=constants.IMAGENET_DEFAULT_MEAN,
            image_std=constants.IMAGENET_DEFAULT_STD,
        )

        logger.info("Loading InceptionV3 model...")
        if (
            embed_model_version is None
            or embed_model_version == "inceptionV3_IMAGENET1K_V1"
        ):
            model = timm_create_model("inception_v3.tv_in1k", pretrained=True)
        else:
            raise ValueError(
                f"Model version {embed_model_version} not supported."
            )

    else:
        processor = CLIPImageProcessor(
            image_mean=constants.IMAGENET_DEFAULT_MEAN,
            image_std=constants.IMAGENET_DEFAULT_STD,
        )

        if embed_model_name == "resnet50":
            logger.info("Loading ResNet50 model...")
            if (
                embed_model_version is None
                or embed_model_version == "resnet50_IMAGENET1K_V2"
            ):
                full_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                full_model_layout = list(full_model.children())
                model = nn_sequential(*full_model_layout[:-1])
            else:
                raise ValueError(
                    f"Model version {embed_model_version} not supported."
                )
        else:
            raise ValueError(f"Model name {embed_model_name} not supported.")

    if cuda_is_available():
        model.to("cuda")
        logger.info("Model loaded to GPU.")
    else:
        logger.info("Model loaded to CPU.")

    if set_eval:
        model.eval()
    return model, processor, embed_model_version


if __name__ == "__main__":
    # Call this module from the root directory of the project
    # via python -m src.helper.load_model --embed_model_name <embed_model_name>
    # for standalone testing purposes
    # e.g. python -m src.helper.load_model --embed_model_name clip

    from PIL import Image

    image_name = "data/val/n01440764/ILSVRC2012_val_00000293.JPEG"
    image = Image.open(image_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str)
    args = parser.parse_args()
    embed_model_name = args.embed_model_name

    model, processor = load_model_and_processor(embed_model_name)
    inputs = processor(images=image, return_tensors="pt")

    if cuda_is_available():
        inputs.pixel_values = inputs.pixel_values.to("cuda")
    print(f"Image Input Shape: {inputs.pixel_values.shape}")

    outputs = model(inputs.pixel_values)
    if embed_model_name == "clip":
        print(f"Image Embeddings Shape: {outputs.image_embeds.shape}")
        print(f"Image Embeddings Sample: {outputs.image_embeds[0][:5]}")
    else:
        print(f"Image Embeddings Shape: {outputs.shape}")
        print(f"Image Embeddings Sample: {outputs[0][:5]}")

    print(
        f"Image Normalization Values: {processor.image_mean},{processor.image_std}"
    )
