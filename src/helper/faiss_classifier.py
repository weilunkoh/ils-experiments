import json

import faiss
from numpy import argmax as np_argmax
from numpy import array as np_array
from numpy import unique as np_unique


class FaissClassifier:
    def __init__(
        self,
        n_neighbors: int = 30,
        index_type: str = "HNSW",
        n_outgoing_links: int = 64,  # Only used by HNSW
    ):
        if index_type not in ["HNSW", "FlatIP", "FlatL2"]:
            raise ValueError(f"Index type {index_type} not supported.")

        self.n_neighbors = n_neighbors
        self.index_type = index_type
        self.n_outgoing_links = n_outgoing_links
        self.fitted = False

    def fit(self, X: np_array, y: np_array):
        self.classes_ = np_unique(y)
        embed_dim = X.shape[1]

        if self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(embed_dim, self.n_outgoing_links)
        elif self.index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(embed_dim)
        elif self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(embed_dim)

        self.training_classes = y
        self.index.add(X)
        self.fitted = True

    def search(self, X: np_array):
        distances, indices = self.index.search(X, self.n_neighbors)
        class_indices = np_array([self.training_classes[i] for i in indices])
        class_predictions = [
            np_unique(class_indices[i], return_counts=True)
            for i in range(len(class_indices))
        ]
        return class_predictions

    def predict(self, X: np_array):
        class_predictions = self.search(X)
        return class_predictions[0][0]

    def predict_proba(self, X: np_array):
        class_predictions = self.search(X)
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

    def save(self, config_path: str, index_path: str):
        if not self.fitted:
            raise ValueError("Classifier must be fitted first.")

        with open(config_path, "w") as file:
            json.dump(
                {
                    "n_neighbors": self.n_neighbors,
                    "embed_dim": self.embed_dim,
                    "index_type": self.index_type,
                    "n_outgoing_links": self.n_outgoing_links,
                    "training_classes": self.training_classes.tolist(),
                },
                file,
            )
        faiss.write_index(self.index, index_path)

    def load(self, config_path: str, index_path: str):
        with open(config_path, "r") as file:
            config = json.load(file)
        self.n_neighbors = config["n_neighbors"]
        self.embed_dim = config["embed_dim"]
        self.index_type = config["index_type"]
        self.n_outgoing_links = config["n_outgoing_links"]
        self.training_classes = np_array(config["training_classes"])
        self.index = faiss.read_index(index_path)
        self.fitted = True
