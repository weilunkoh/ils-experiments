import argparse
import logging
from json import dump as json_dump
from os import makedirs as os_makedirs
from os.path import exists as path_exists

import torchvision.datasets.imagenet as imagenet_py
from torchvision.datasets import ImageNet

from src.helper.modified.torchvision_dataset_utils import download_url

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# Original imagenet_py.ARCHIVE_META from torchvision 0.15.0
# imagenet_pyARCHIVE_META = {
#     "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
#     "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
#     "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
# }

# Modified (Check the logs for md5 checksum error and update below values accordingly.)
imagenet_py.ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": (
        "ILSVRC2012_devkit_t12.tar.gz",
        "6e6ed663724924617097c22e2dd978f4",
    ),
}


def download_images(split: str) -> None:
    """Download images from ImageNet dataset.

    Args:
        split (str): Split of the dataset. Can be either 'train' or 'val'.
    """
    if split not in ["train", "val"]:
        raise ValueError("Split must be either 'train' or 'val'.")

    if not path_exists("data"):
        os_makedirs("data")

    # Download imagenet files separately because torchvision.datasets.ImageNet
    # checks against an old md5 hash of the imagenet files keep failing.
    imagenet_url_suffix = "https://image-net.org/data/ILSVRC/2012"

    # Save meta gz file
    meta_zipname = "ILSVRC2012_devkit_t12.tar.gz"
    if not path_exists(f"data/{meta_zipname}"):
        logger.info("Downloading meta gz file...")
        meta_md5 = imagenet_py.ARCHIVE_META["devkit"][1]
        meta_url = f"{imagenet_url_suffix}/{meta_zipname}"
        download_url(meta_url, "data", md5=meta_md5)
    else:
        logger.info("Meta gz file already exists.")

    # Save img tar file
    img_tar_filename = f"ILSVRC2012_img_{split}.tar"
    if not path_exists(f"data/{img_tar_filename}"):
        logger.info(f"Downloading img tar file for {split} split...")
        if split == "train":
            img_tar_md5 = imagenet_py.ARCHIVE_META["train"][1]
        else:
            img_tar_md5 = imagenet_py.ARCHIVE_META["val"][1]
        img_tar_url = f"{imagenet_url_suffix}/{img_tar_filename}"
        download_url(img_tar_url, "data", md5=img_tar_md5)
    else:
        logger.info(f"Img tar file for {split} split already exists.")

    # Download images via Torchvision's ImageNet dataset class.
    logging.info(f"Downloading {split} images...")
    dataset = ImageNet(root="data", split=split)
    logging.info(f"Downloaded {split} images...")

    # Save wnid to class mapping
    class_mapping = {}
    for wnid in dataset.wnid_to_idx:
        class_mapping[wnid] = dataset.classes[dataset.wnid_to_idx[wnid]]
    with open(f"data/{split}_class_mapping.json", "w") as f:
        json_dump(class_mapping, f, indent=4)
    logging.info(f"Saved {split} class mapping.")


if __name__ == "__main__":
    # Call this module from the root directory of the project
    # via python -m scripts.download_images --split <split_name>
    # e.g. python -m scripts.download_images --split train

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    args = parser.parse_args()

    download_images(args.split)
