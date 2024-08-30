import logging
from json import dump as json_dump
from json import load as json_load
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os.path import exists as path_exists
from os.path import join as path_join
from os.path import split as path_split

import torch
from numpy import argmax as np_argmax
from numpy import array as np_array
from numpy import unique as np_unique
from torch.utils.data import DataLoader, Dataset


class TensorDatasetWithClass(Dataset):
    def __init__(self, file_paths, class_names):
        self.file_paths = file_paths
        self.class_names = class_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tensor = torch.load(self.file_paths[idx])
        class_name = self.class_names[idx]
        return tensor, class_name


class KNNTorchClassifier:
    def __init__(
        self,
        n_neighbors: int = 10,
        batch_size: int = 1500000,
        save_jsons: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.save_jsons = save_jsons
        self.fitted = False

    def fit(self, source_folder: str, y: np_array = None):
        self.source_folder = source_folder
        if y is not None:
            self.classes_ = np_unique(y)

        folder_names = os_listdir(source_folder)
        if "json" in folder_names:
            folder_names.remove("json")

        file_paths = []
        class_names = []

        for class_name in folder_names:
            class_path = path_join(source_folder, class_name)
            for file_name in os_listdir(class_path):
                file_path = path_join(class_path, file_name)
                file_paths.append(file_path)
                class_names.append(class_name)

        if len(class_names) < self.batch_size:
            self.num_batches = 1
        elif len(class_names) % self.batch_size > 0:
            self.num_batches = len(class_names) // self.batch_size + 1
        else:
            self.num_batches = len(class_names) // self.batch_size

        self.tensor_dataset = TensorDatasetWithClass(file_paths, class_names)
        self.data_loader = DataLoader(
            self.tensor_dataset, batch_size=self.batch_size, shuffle=False
        )
        if self.num_batches == 1:
            logging.info("Pre-loading all images as one batch")
            for batch, class_names in self.data_loader:
                self.batch = batch
                self.batch_norm = batch / batch.norm(dim=1, keepdim=True)

        self.fitted = True

    def search(self, filenames: list, folder: str):
        if not self.fitted:
            raise ValueError("Classifier must be fitted first.")

        if not path_exists(f"{folder}/knn_all"):
            os_makedirs(f"{folder}/knn_all")

        class_predictions = []
        count = 1
        top_n = self.n_neighbors
        if self.num_batches > 1:
            all_top_similarities = {}
            all_top_indices = {}
            all_top_class_names = {}
            current_index = 0

            count = 1
            for batch, class_names in self.data_loader:
                if "cuda" not in str(batch.device):
                    batch = batch.cuda()
                batch_norm = batch / batch.norm(dim=1, keepdim=True)

                for filename in filenames:
                    all_top_similarities[filename] = all_top_similarities.get(
                        filename, []
                    )
                    all_top_indices[filename] = all_top_indices.get(
                        filename, []
                    )
                    all_top_class_names[filename] = all_top_class_names.get(
                        filename, []
                    )

                    # Get the target tensor from file
                    image_path = path_join(folder, filename)
                    target_tensor = torch.load(image_path)
                    if "cuda" not in str(target_tensor.device):
                        target_tensor = target_tensor.cuda()

                    # Normalize the target tensor
                    target_tensor_norm = target_tensor / target_tensor.norm(
                        dim=0, keepdim=True
                    )

                    # Compute cosine similarity
                    cosine_similarities = torch.matmul(
                        batch_norm, target_tensor_norm
                    )

                    # Concatenate current batch similarities and indices with existing ones
                    combined_similarities = torch.cat(
                        (
                            torch.tensor(
                                all_top_similarities[filename]
                            ).cuda(),
                            cosine_similarities,
                        )
                    )
                    combined_indices = torch.cat(
                        (
                            torch.tensor(all_top_indices[filename]).cuda(),
                            torch.arange(
                                current_index, current_index + self.batch_size
                            ).cuda(),
                        )
                    )
                    combined_class_names = all_top_class_names[
                        filename
                    ] + list(class_names)

                    # Get top-n similarities and indices
                    top_values, top_indices_in_combined = torch.topk(
                        combined_similarities, top_n, largest=True, sorted=True
                    )
                    all_top_indices[filename] = combined_indices[
                        top_indices_in_combined
                    ].tolist()
                    all_top_similarities[filename] = top_values.tolist()
                    all_top_class_names[filename] = [
                        combined_class_names[i]
                        for i in top_indices_in_combined.tolist()
                    ]

                current_index += self.batch_size
                logging.info(
                    f"Completed class prediction for batch {count}/{self.num_batches}"
                )
                count += 1

            class_predictions = [
                np_unique(
                    np_array(all_top_class_names[filename]), return_counts=True
                )
                for filename in filenames
            ]

            if self.save_jsons:
                count = 1
                num_files = len(filenames)
                for filename in filenames:
                    json_filename = filename.replace(".pt", ".json").split(
                        "/"
                    )[-1]
                    filepaths = [
                        path_split(self.tensor_dataset.file_paths[int(i)])[
                            -1
                        ].replace(".pt", "")
                        for i in all_top_indices[filename]
                    ]
                    json_to_save = {
                        filepath: {
                            "name": class_name,
                            "similarity": similarity,
                        }
                        for filepath, class_name, similarity in zip(
                            filepaths,
                            all_top_class_names[filename],
                            all_top_similarities[filename],
                        )
                    }
                    with open(
                        f"{folder}/knn_all/{json_filename}", "w"
                    ) as file:
                        json_dump(
                            json_to_save,
                            file,
                            indent=4,
                        )
                    logging.info(
                        f"Saved JSON for {filename} ({count}/{num_files})"
                    )
                    count += 1

        else:
            for filename in filenames:
                # Get the target tensor from file
                image_path = path_join(folder, filename)
                target_tensor = torch.load(image_path)
                if "cuda" not in str(target_tensor.device):
                    target_tensor = target_tensor.cuda()

                # Normalize the target tensor
                target_tensor_norm = target_tensor / target_tensor.norm(
                    dim=0, keepdim=True
                )

                cosine_similarities = torch.matmul(
                    self.batch_norm, target_tensor_norm
                )
                top_values, top_indices = torch.topk(
                    cosine_similarities, top_n, largest=True, sorted=True
                )
                top_class_names = [
                    self.tensor_dataset.class_names[i]
                    for i in top_indices.tolist()
                ]

                class_prediction = np_unique(
                    np_array(top_class_names), return_counts=True
                )
                class_predictions.append(class_prediction)
                logging.info(
                    f"Completed class prediction for {filename} ({count}/{len(filenames)})"
                )
                count += 1

                if self.save_jsons:
                    json_filename = filename.replace(".pt", ".json").split(
                        "/"
                    )[-1]
                    filepaths = [
                        path_split(self.tensor_dataset.file_paths[i])[
                            -1
                        ].replace(".pt", "")
                        for i in top_indices
                    ]
                    json_to_save = {
                        filepath: {
                            "name": class_name,
                            "similarity": similarity,
                        }
                        for filepath, class_name, similarity in zip(
                            filepaths,
                            top_class_names,
                            top_values.tolist(),
                        )
                    }
                    with open(
                        f"{folder}/knn_all/{json_filename}", "w"
                    ) as file:
                        json_dump(
                            json_to_save,
                            file,
                            indent=4,
                        )

        return class_predictions

    def predict_proba(self, filenames: list, folder: str):
        logging.info(f"Predicting probabilities for {len(filenames)} files...")
        actual_batch_size = min(self.batch_size, len(self.tensor_dataset))
        logging.info(f"Using {self.num_batches} batches")
        logging.info(f"{actual_batch_size} training images per batch")

        class_predictions = self.search(filenames=filenames, folder=folder)
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
                    "n_neighbors": self.n_neighbors,
                    "batch_size": self.batch_size,
                    "classes_": self.classes_.tolist(),
                    "source_folder": self.source_folder,
                },
                file,
                indent=4,
            )

    def load(self, config_path: str):
        with open(config_path, "r") as file:
            config = json_load(file)

        self.n_neighbors = config["n_neighbors"]
        self.batch_size = config["batch_size"]
        self.classes_ = np_array(config["classes_"])
        self.fit(config["source_folder"])
