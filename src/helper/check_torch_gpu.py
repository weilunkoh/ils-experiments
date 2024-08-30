import logging

from torch.cuda import get_device_name, is_available

logger = logging.getLogger()


def check_torch_gpu():
    # Call this module from the root directory of the project
    # via python -m src.helper.check_torch_gpu for standalone testing purposes
    # e.g. python -m src.helper.check_torch_gpu

    if is_available():
        logger.info(f"GPU: {get_device_name()}")
    else:
        logger.info("GPU: Not available")


if __name__ == "__main__":
    check_torch_gpu()
