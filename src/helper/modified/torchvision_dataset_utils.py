import logging
import os
import os.path
import urllib
from time import gmtime, localtime, strftime, time
from typing import Iterator, Optional

from torchvision._internally_replaced_utils import (
    _download_file_from_remote_location,
    _is_remote_location_available,
)
from torchvision.datasets.utils import (
    USER_AGENT,
    _get_google_drive_file_id,
    _get_redirect_url,
    calculate_md5,
    check_integrity,
    download_file_from_google_drive,
)

logger = logging.getLogger()

# Modified from torchvision to not use tqdm to prevent the progress bar from
# creating a long list of logs in the resulting output file when running a
# a sbatch job in crimson server cluster.


def fmt_time(seconds: float, elapsed_time=False) -> str:
    if elapsed_time:
        return strftime("%H:%M:%S", gmtime(seconds))
    else:
        return strftime("%d/%m/%Y %H:%M:%S", localtime(seconds))


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    chunk_size: int,
    length: int,
    update_interval: float = 0.1,
) -> None:
    start_time = time()
    logger.info(
        f"Downloading to {destination} started at {fmt_time(start_time)}."
    )
    next_interval = update_interval
    num_chunks = length // chunk_size + 1
    next_chunk = next_interval * num_chunks
    with open(destination, "wb") as fh:
        chunk_idx = 1
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            if chunk_idx >= next_chunk:
                downloaded_bytes = chunk_idx * chunk_size
                msg1 = f"Downloaded {downloaded_bytes} bytes out of {length} bytes."
                elapsed_time = time() - start_time
                msg2 = f"Elapsed time: {fmt_time(elapsed_time, elapsed_time=True)}."
                logger.info(f"{msg1} ({next_interval*100:.2f}%) {msg2}")
                next_interval += update_interval
                next_chunk = next_interval * num_chunks
            chunk_idx += 1

        if next_interval < 1:
            elapsed_time = time() - start_time
            msg = f"Downloaded 100.00%. Elapsed time: {fmt_time(elapsed_time, elapsed_time=True)}."
            logger.info(msg)


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            chunk_size=chunk_size,
            length=response.length,
        )


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    if _is_remote_location_available():
        _download_file_from_remote_location(fpath, url)
    else:
        # expand redirect chain if needed
        url = _get_redirect_url(url, max_hops=max_redirect_hops)

        # check if file is located on Google Drive
        file_id = _get_google_drive_file_id(url)
        if file_id is not None:
            return download_file_from_google_drive(
                file_id, root, filename, md5
            )

        # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            _urlretrieve(url, fpath)
        except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead. Downloading "
                    + url
                    + " to "
                    + fpath
                )
                _urlretrieve(url, fpath)
            else:
                raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        logger.info(f"Update md5 value to {calculate_md5(fpath)} for {fpath}")
        raise RuntimeError("File not found or corrupted.")
