import logging
from json import dump as json_dump
from json import load as json_load
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists
from threading import Thread

from numpy import argmax as np_argmax
from numpy import array as np_array
from numpy import dot as np_dot
from numpy import unique as np_unique
from numpy.linalg import norm as np_norm


class KNNAllClassifier:
    def __init__(
        self,
        n_neighbors: int = 200,
        num_search_threads: int = 20,
        num_compare_class_threads: int = 20,
        num_cos_sim_threads: int = 20,
    ):
        self.n_neighbors = n_neighbors
        self.num_search_threads = num_search_threads
        self.num_compare_class_threads = num_compare_class_threads
        self.num_cos_sim_threads = num_cos_sim_threads
        self.fitted = False

    def fit(self, source_folder: str, y: np_array):
        self.source_folder = source_folder
        self.classes_ = np_unique(y)
        self.fitted = True
        logging.info(
            f"KNNAllClassifier fitted with {len(self.classes_)} classes."
        )

    def calc_cos_sim(
        self,
        current_img_features: np_array,
        train_img_features_dict: dict,
        img_filename: str,
        class_name: str,
        current_class_nearest: dict,
    ):
        train_img_features = train_img_features_dict[img_filename]
        train_img_features = np_array(train_img_features)

        # Compute cosine similarity
        cos_sim = np_dot(current_img_features, train_img_features) / (
            np_norm(current_img_features) * np_norm(train_img_features)
        )
        current_class_nearest[img_filename] = {
            "name": class_name,
            "similarity": cos_sim,
        }

    def make_nn_json_for_one_class(
        self,
        filepath: str,
        current_img_features: np_array,
        json_file: str,
        every_class_nearest: dict,
    ):
        current_class_nearest = {}

        with open(f"{self.source_folder}/{json_file}", "r") as file:
            train_img_features_dict = json_load(file)
        class_name = json_file.replace("_features.json", "")

        if self.num_cos_sim_threads > 1:
            cos_sim_batches = [
                list(train_img_features_dict.keys())[
                    i : i + self.num_cos_sim_threads
                ]
                for i in range(
                    0, len(train_img_features_dict), self.num_cos_sim_threads
                )
            ]

            for cos_sim_batch in cos_sim_batches:
                cos_sim_threads = []
                for img_filename in cos_sim_batch:
                    cos_sim_thread = Thread(
                        target=self.calc_cos_sim,
                        args=(
                            current_img_features,
                            train_img_features_dict,
                            img_filename,
                            class_name,
                            current_class_nearest,
                        ),
                    )
                    cos_sim_threads.append(cos_sim_thread)

                for current_thread in cos_sim_threads:
                    current_thread.start()
                for current_thread in cos_sim_threads:
                    current_thread.join()
        else:
            for img_filename in train_img_features_dict:
                self.calc_cos_sim(
                    current_img_features,
                    train_img_features_dict,
                    img_filename,
                    class_name,
                    current_class_nearest,
                )

        # Sort current_class_nearest by similarity
        current_class_nearest = {
            item[0]: item[1]
            for item in sorted(
                current_class_nearest.items(),
                key=lambda x: x[1]["similarity"],
                reverse=True,
            )
        }

        # Add only the top n_neighbors to every_class_nearest
        current_class_nearest = dict(
            list(current_class_nearest.items())[: self.n_neighbors]
        )

        for key in current_class_nearest:
            every_class_nearest[key] = current_class_nearest[key]

    def make_nn_json(self, x: list, filepath: str):
        current_img_features = np_array(x)

        every_class_nearest = {}
        json_files = os_listdir(self.source_folder)

        if self.num_compare_class_threads > 1:
            # Split json files into batches of num_compare_class_threads
            json_files_batches = [
                json_files[i : i + self.num_compare_class_threads]
                for i in range(
                    0, len(json_files), self.num_compare_class_threads
                )
            ]

            for json_file_batch in json_files_batches:
                compare_class_threads = []
                for json_file in json_file_batch:
                    compare_class_thread = Thread(
                        target=self.make_nn_json_for_one_class,
                        args=(
                            filepath,
                            current_img_features,
                            json_file,
                            every_class_nearest,
                        ),
                    )
                    compare_class_threads.append(compare_class_thread)

                for current_thread in compare_class_threads:
                    current_thread.start()
                for current_thread in compare_class_threads:
                    current_thread.join()

                # Sort every_class_nearest by similarity
                every_class_nearest = {
                    item[0]: item[1]
                    for item in sorted(
                        every_class_nearest.items(),
                        key=lambda x: x[1]["similarity"],
                        reverse=True,
                    )
                }

                # Keep only the top n_neighbors in every_class_nearest
                every_class_nearest = dict(
                    list(every_class_nearest.items())[: self.n_neighbors]
                )
        else:
            for json_file in json_files:
                self.make_nn_json_for_one_class(
                    filepath,
                    current_img_features,
                    json_file,
                    every_class_nearest,
                )

                # Sort every_class_nearest by similarity
                every_class_nearest = {
                    item[0]: item[1]
                    for item in sorted(
                        every_class_nearest.items(),
                        key=lambda x: x[1]["similarity"],
                        reverse=True,
                    )
                }

                # Keep only the top n_neighbors in every_class_nearest
                every_class_nearest = dict(
                    list(every_class_nearest.items())[: self.n_neighbors]
                )
        return every_class_nearest

    def search_for_one_img(
        self,
        x: list,
        filename: str,
        filepath: str,
        class_predictions_dict: dict,
    ):
        logging.info(f"Starting search thread for {filepath}.")
        if path_exists(filepath):
            with open(filepath, "r") as file:
                nearest_neighbors = json_load(file)
            if len(nearest_neighbors) < self.n_neighbors:
                nearest_neighbors = self.make_nn_json(x, filepath)
                with open(filepath, "w") as fp:
                    json_dump(nearest_neighbors, fp, indent=4)
        else:
            nearest_neighbors = self.make_nn_json(x, filepath)
            with open(filepath, "w") as fp:
                json_dump(nearest_neighbors, fp, indent=4)

        # Sort the nearest neighbours by similarity
        class_names = np_array(
            [
                k["name"]
                for k in sorted(
                    nearest_neighbors.values(),
                    key=lambda x: x["similarity"],
                    reverse=True,
                )
            ]
        )
        class_names = class_names[: self.n_neighbors]
        x_class_predictions = np_unique(class_names, return_counts=True)
        class_predictions_dict[filename] = x_class_predictions

    def search(self, X: list, filenames: list = None, folder: str = None):
        if not self.fitted:
            raise ValueError("Classifier must be fitted first.")

        if not path_exists(f"{folder}/knn_all"):
            os_makedirs(f"{folder}/knn_all")

        class_predictions_dict = {}

        logging.info(f"Performing search for {len(filenames)} images.")

        if self.num_search_threads > 1:
            search_threads = []

            for x, filename in zip(X, filenames):
                filepath = f"{folder}/knn_all/{filename}.json"
                search_thread = Thread(
                    target=self.search_for_one_img,
                    args=(x, filename, filepath, class_predictions_dict),
                )
                search_threads.append(search_thread)

            # Split search threads into groups of num_search_threads
            for i in range(0, len(search_threads), self.num_search_threads):
                search_threads_group = search_threads[
                    i : i + self.num_search_threads
                ]
                for search_thread in search_threads_group:
                    search_thread.start()
                for search_thread in search_threads_group:
                    search_thread.join()
        else:
            for x, filename in zip(X, filenames):
                filepath = f"{folder}/knn_all/{filename}.json"
                self.search_for_one_img(
                    x, filename, filepath, class_predictions_dict
                )

        class_predictions = [
            class_predictions_dict[filename] for filename in filenames
        ]
        return class_predictions

    def predict(self, X: list, filenames: list = None, folder: str = None):
        class_predictions = self.search(X, filenames=filenames, folder=folder)
        return class_predictions[0][0]

    def predict_proba(
        self, X: list, filenames: list = None, folder: str = None
    ):
        class_predictions = self.search(X, filenames=filenames, folder=folder)
        class_predictions = [
            (class_names, class_counts / self.n_neighbors)
            for class_names, class_counts in class_predictions
        ]
        overall_pred_probas = []
        for class_names, class_probas in class_predictions:
            pred_probas = [0] * len(self.classes_)
            for class_name, class_proba in zip(class_names, class_probas):
                class_index = np_argmax(self.classes_ == class_name)
                pred_probas[class_index] = class_proba
            overall_pred_probas.append(pred_probas)

        overall_pred_probas = np_array(overall_pred_probas)
        return overall_pred_probas

    def save(self, config_path: str):
        if not self.fitted:
            raise ValueError("Classifier must be fitted first.")

        with open(config_path, "w") as file:
            json_dump(
                {
                    "classes_": self.classes_.tolist(),
                    "n_neighbors": self.n_neighbors,
                },
                file,
            )

    def load(self, config_path: str):
        with open(config_path, "r") as file:
            config = json_load(file)

        self.classes_ = np_array(config["classes_"])
        self.n_neighbors = config["n_neighbors"]
        self.fitted = True
