import argparse
import logging
from json import dump as json_dump
from json import load as json_load
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists
from typing import Union

from numpy import array as np_array
from sklearn.cluster import KMeans, MiniBatchKMeans

from scripts.extract_features import extract_features
from src.helper.common_values import CENTROID_INIT_METHODS, CLUSTERING_METHODS

logger = logging.getLogger()


def split_list(input_list: list, split_fraction: float) -> list:
    # Calculate the number of elements in each sublist should have
    sublist_length = int(len(input_list) * split_fraction)
    sublist_lengths = []
    total_length = len(input_list)

    while sum(sublist_lengths) < total_length:
        sublist_lengths.insert(0, sublist_length)
        if total_length - sum(sublist_lengths) < sublist_length:
            sublist_lengths[0] += total_length - sum(sublist_lengths)

    # Create sublists
    sublists = []
    start_index = 0
    for current_length in sublist_lengths:
        end_index = start_index + current_length
        sublists.append(input_list[start_index:end_index])
        start_index = end_index

    logger.info([len(sublist) for sublist in sublists])
    return sublists


def compute_centroids(
    embed_model_name: str,
    data_folder_name: str,
    clustering_method: str,
    num_centroids: int,
    centroids_initialisation: str,
    max_iter: int,
    random_state: int,
    embed_model_version: str = None,
    incremental_step_size: Union[int, None] = None,
    abs_path_prefix: str = "",
    output_already_logs: bool = True,
    add_stream_handler: bool = True,
) -> None:
    if add_stream_handler:
        logger.addHandler(logging.StreamHandler())

    # Check if extracted features exist. If not, extract them.
    embed_model_version = extract_features(
        embed_model_name,
        data_folder_name,
        embed_model_version=embed_model_version,
        abs_path_prefix=abs_path_prefix,
        output_already_logs=False,
        add_stream_handler=False,
    )

    if clustering_method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Clustering_method must be one of {CLUSTERING_METHODS}"
        )

    if clustering_method == "kmeans":
        if centroids_initialisation not in CENTROID_INIT_METHODS:
            raise ValueError(
                f"Centroids_initialisation for kmeans must be one of {CENTROID_INIT_METHODS}"
            )

        if centroids_initialisation == "incremental":
            if incremental_step_size is None:
                raise ValueError(
                    "incremental_step_size must be specified for kmeans with incremental"
                )
    elif clustering_method == "minibatchkmeans":
        if centroids_initialisation != "k-means++":
            raise ValueError(
                "Centroids_initialisation for minibatchkmeans must be k-means++"
            )

        if incremental_step_size is None:
            raise ValueError(
                "incremental_step_size must be specified for minibatchkmeans"
            )

    # Get features folder and associated json files
    source_folder_1 = f"{abs_path_prefix}/data/features"
    source_folder_2 = f"{embed_model_version}/{data_folder_name}"
    source_folder = f"{source_folder_1}/{source_folder_2}"
    source_json_folder = f"{source_folder}/json"

    # Make target folder for centroids
    base_prefix = f"{abs_path_prefix}/data/centroids"
    target_folder_0 = f"{base_prefix}/{embed_model_version}"
    target_folder_1 = f"{target_folder_0}/{data_folder_name}"
    target_folder_2 = f"rs_{random_state}/{clustering_method}"
    target_folder_3 = f"num_{num_centroids}/{centroids_initialisation}"
    if centroids_initialisation == "incremental":
        target_folder_3 = f"{target_folder_3}_{incremental_step_size}"
    if clustering_method == "minibatchkmeans":
        target_folder_3 = (
            f"{target_folder_3}/incremental_{incremental_step_size}"
        )
    target_folder = f"{target_folder_1}/{target_folder_2}/{target_folder_3}"
    os_makedirs(target_folder, exist_ok=True)

    for json_file in os_listdir(source_json_folder):
        target_folder_json = f"{target_folder}/{json_file}"

        if not path_exists(target_folder_json):
            logger.info(f"Computing centroids for {json_file}...")
            features_json = json_load(
                open(f"{source_json_folder}/{json_file}", "r")
            )
            features_list = [v for v in features_json.values()]
            features = np_array(features_list)

            if clustering_method == "kmeans":
                if centroids_initialisation == "k-means++":
                    clustering_model = KMeans(
                        n_clusters=num_centroids,
                        init=centroids_initialisation,
                        n_init=1,
                        max_iter=max_iter,
                        random_state=random_state,
                    )
                    clustering_model.fit(features)
                elif centroids_initialisation == "incremental":
                    # Split features into sublists
                    features_sublists = split_list(
                        input_list=features_list,
                        split_fraction=incremental_step_size,
                    )

                    current_init = "k-means++"
                    for idx, sublist in enumerate(features_sublists):
                        logger.info(
                            f"Sublist {idx+1} of {len(features_sublists)}"
                        )
                        clustering_model = KMeans(
                            n_clusters=num_centroids,
                            init=current_init,
                            n_init=1,
                            max_iter=max_iter,
                            random_state=random_state,
                        )
                        clustering_model.fit(sublist)
                        current_init = clustering_model.cluster_centers_
            elif clustering_method == "minibatchkmeans":
                batch_size = int(len(features_list) * incremental_step_size)
                clustering_model = MiniBatchKMeans(
                    n_clusters=num_centroids,
                    init=centroids_initialisation,
                    n_init=1,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    random_state=random_state,
                )
                clustering_model.fit(features)

            centroids = clustering_model.cluster_centers_
            centroids_record = centroids.tolist()
            with open(target_folder_json, "w") as f:
                json_dump(centroids_record, f, indent=4)

            logger.info(f"Computed centroids for {json_file}.")
        else:
            if output_already_logs:
                logger.info(f"Centroids for {json_file} already exist.")

    return target_folder


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.compute_centroids \
    # --embed_model_name <embed_model_name> --data_folder_name <data_folder_name> \
    # --clustering_method <clustering_method> \
    # --centroids_initialisation <centroids_initialisation> \
    # --num_centroids <num_centroids> --random_state <random_state> \

    # e.g.:
    # python -m scripts.compute_centroids \
    # --embed_model_name clip --data_folder_name train \
    # --clustering_method kmeans --centroids_initialisation k-means++ \
    # --num_centroids 30 --random_state 42 \

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model_name", type=str)
    parser.add_argument("--data_folder_name", type=str)
    parser.add_argument("--clustering_method", type=str)
    parser.add_argument("--num_centroids", type=int)
    parser.add_argument("--centroids_initialisation", type=str)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--random_state", type=int)
    args = parser.parse_args()

    embed_model_name = args.embed_model_name
    data_folder_name = args.data_folder_name
    num_centroids = args.num_centroids
    clustering_method = args.clustering_method
    centroids_initialisation = args.centroids_initialisation
    max_iter = args.max_iter
    random_state = args.random_state
    compute_centroids(
        embed_model_name,
        data_folder_name,
        clustering_method,
        num_centroids,
        centroids_initialisation,
        max_iter,
        random_state,
    )
