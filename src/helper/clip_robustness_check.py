from PIL.JpegImagePlugin import JpegImageFile
from torch import no_grad
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionModelWithProjection,
)

from src.helper.load_model import load_model_and_processor

clip_version = "clip-vit-large-patch14"


def load_clip_text():
    text_model = CLIPTextModelWithProjection.from_pretrained(
        f"openai/{clip_version}"
    )
    tokenizer = CLIPTokenizerFast.from_pretrained(f"openai/{clip_version}")

    # Check for GPU
    if cuda_is_available():
        text_model.to("cuda")

    return text_model, tokenizer


def load_clip_models_and_processor():
    image_model, image_processor, _ = load_model_and_processor(
        "clip", embed_model_version=clip_version
    )
    text_model, tokenizer = load_clip_text()
    return image_model, image_processor, text_model, tokenizer


def clip_cosine_similarity(
    clip_image_model: CLIPVisionModelWithProjection,
    clip_image_processor: CLIPImageProcessor,
    clip_text_model: CLIPTextModelWithProjection,
    clip_tokenizer: CLIPTokenizerFast,
    device: str,
    pil_image: JpegImageFile,
    class_name: str,
) -> bool:
    # Get image features
    image_inputs = clip_image_processor(images=pil_image, return_tensors="pt")
    image_inputs.pixel_values = image_inputs.pixel_values.to(device)
    with no_grad():
        image_outputs = clip_image_model(image_inputs.pixel_values)
        image_features = image_outputs.image_embeds

    # Get text features of class name
    text_inputs = clip_tokenizer([class_name], return_tensors="pt")
    text_inputs.input_ids = text_inputs.input_ids.to(device)
    text_inputs.attention_mask = text_inputs.attention_mask.to(device)
    with no_grad():
        text_outputs = clip_text_model(
            text_inputs.input_ids, text_inputs.attention_mask
        )
        text_features = text_outputs.text_embeds

    # Check if image and text features are similar
    # by calculating the cosine similarity
    similarity_score = cosine_similarity(image_features, text_features).item()
    return similarity_score
