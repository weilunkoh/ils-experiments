import logging
from json import dump as json_dump
from json import load as json_load
from os import getcwd as os_getcwd
from os import listdir as os_listdir
from os import makedirs as os_makedirs

from hydra import main as hydra_main
from hydra.utils import get_original_cwd, to_absolute_path
from joblib import dump as joblib_dump
from numpy import array as np_array
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from scripts.compute_centroids import compute_centroids
from scripts.extract_features import extract_features
from src.helper import common_values
from src.helper.faiss_classifier import FaissClassifier
from src.helper.knn_all_classifier import KNNAllClassifier
from src.helper.knn_torch_classifier import KNNTorchClassifier
from src.helper.prob_zero_handler import handle_zero

# Use hydra logging instead of custom logging
logger = logging.getLogger(__name__)


@hydra_main(
    version_base=None, config_path="../config", config_name="experiment.yaml"
)
def run_experiment(cfg: DictConfig) -> None:
    # Saving a copy of experiment details.
    # Note: This line works most safely when running one experiment at a time.
    original_cfg = OmegaConf.load(to_absolute_path("config/experiment.yaml"))
    os_makedirs(f"{os_getcwd()}/config")
    with open(f"{os_getcwd()}/config/experiment.yaml", "w") as file:
        OmegaConf.save(original_cfg, file)

    # Check if configuration values are allowed
    if cfg.experiment.embed_model not in common_values.EMBED_MODEL_NAMES:
        err_cfg_name = "cfg.experiment.embed_model"
        raise ValueError(
            f"{err_cfg_name} must be one of {common_values.EMBED_MODEL_NAMES}."
        )
    else:
        logger.info(
            f"Passed check for embedding model: {cfg.experiment.embed_model}"
        )

    allowed_classifier_names = common_values.CENTROID_CLASSIFIER_METHODS
    allowed_classifier_names.extend(common_values.CLASSIFIER_METHODS)
    allowed_classifier_names = list(set(allowed_classifier_names))
    if cfg.experiment.classifier not in allowed_classifier_names:
        err_cfg_name = "cfg.experiment.classifier"
        raise ValueError(
            f"{err_cfg_name} must be one of {allowed_classifier_names}."
        )
    else:
        logger.info(
            f"Passed check for classifier: {cfg.experiment.classifier}"
        )

    # Check for computed centroids if needed. If not, just check for train features.
    if cfg.experiment.use_centroids:
        if (
            cfg.experiment.centroids_cluster
            not in common_values.CLUSTERING_METHODS
        ):
            err_cfg_name = "cfg.experiment.centroids_cluster"
            raise ValueError(
                f"{err_cfg_name} must be one of {common_values.CLUSTERING_METHODS}."
            )
        else:
            logger.info(
                f"Passed check for centroids_cluster: {cfg.experiment.centroids_cluster}"
            )
        if (
            cfg.experiment.kmeans.centroids_init
            not in common_values.CENTROID_INIT_METHODS
        ):
            err_cfg_name = "cfg.experiment.kmeans.centroids_init"
            raise ValueError(
                f"{err_cfg_name} must be one of {common_values.CENTROID_INIT_METHODS}."
            )
        else:
            logger.info(
                f"Passed check for centroids_init: {cfg.experiment.kmeans.centroids_init}"
            )
        source_folder = compute_centroids(
            cfg.experiment.embed_model,
            cfg.experiment.train_data_folder,
            cfg.experiment.centroids_cluster,
            cfg.experiment.kmeans.num_centroids,
            cfg.experiment.kmeans.centroids_init,
            cfg.experiment.kmeans.max_iter,
            cfg.experiment.seed,
            embed_model_version=cfg.experiment.embed_model_version,
            incremental_step_size=cfg.experiment.kmeans.incremental_step,
            abs_path_prefix=get_original_cwd(),
            output_already_logs=False,
            add_stream_handler=False,
        )
    else:
        if cfg.experiment.classifier not in common_values.CLASSIFIER_METHODS:
            classifier_names = "faiss or custom knn (i.e. knn_all)"
            err_msg = f"Only {classifier_names} classifier supports not using centroids."
            raise ValueError(err_msg)

        extract_features(
            cfg.experiment.embed_model,
            cfg.experiment.train_data_folder,
            embed_model_version=cfg.experiment.embed_model_version,
            abs_path_prefix=get_original_cwd(),
            output_already_logs=False,
            add_stream_handler=False,
        )
        embed_model_version = cfg.experiment.embed_model_version
        embed_folder = f"data/features/{embed_model_version}"
        train_folder = cfg.experiment.train_data_folder
        rel_source_folder = f"{embed_folder}/{train_folder}/json"
        source_folder = to_absolute_path(rel_source_folder)

    # Check to extract features for eval set.
    # extract_features(
    #     cfg.experiment.embed_model,
    #     cfg.experiment.eval_data_folder,
    #     embed_model_version=cfg.experiment.embed_model_version,
    #     abs_path_prefix=get_original_cwd(),
    #     output_already_logs=False,
    #     add_stream_handler=False,
    # )

    # Create folder to store artifacts such as models and results
    target_folder = f"{os_getcwd()}/artifacts"
    os_makedirs(target_folder)

    # Fit classifier
    fit_msg = f"Fitting {cfg.experiment.classifier} classifier..."
    if cfg.experiment.classifier == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=cfg.experiment.knn.num_neighbors,
        )
    elif cfg.experiment.classifier == "logreg":
        # Note: Default solver does not require random state to be set
        classifier = LogisticRegression(
            max_iter=cfg.experiment.logreg.max_iter,
        )
    elif cfg.experiment.classifier == "faiss":
        classifier = FaissClassifier(
            n_neighbors=cfg.experiment.faiss.num_neighbors,
            index_type=cfg.experiment.faiss.index_type,
            n_outgoing_links=cfg.experiment.faiss.num_outgoing_links,
        )
        if not cfg.experiment.use_centroids:
            fit_msg = "Fitting faiss classifier without centroids..."
    elif cfg.experiment.classifier == "knn_all":
        classifier = KNNAllClassifier(
            n_neighbors=cfg.experiment.knn_all.num_neighbors,
            num_search_threads=cfg.experiment.knn_all.num_search_threads,
            num_compare_class_threads=cfg.experiment.knn_all.num_compare_class_threads,
            num_cos_sim_threads=cfg.experiment.knn_all.num_cos_sim_threads,
        )
        fit_msg = "Fitting knn classifier without centroids..."
    elif cfg.experiment.classifier == "knn_torch":
        classifier = KNNTorchClassifier(
            n_neighbors=cfg.experiment.knn_torch.num_neighbors,
            batch_size=cfg.experiment.knn_torch.batch_size,
            save_jsons=cfg.experiment.knn_torch.save_jsons,
        )
        fit_msg = "Fitting knn torch classifier without centroids..."

    logger.info(fit_msg)

    if cfg.experiment.classifier not in ["knn_all", "knn_torch"]:
        X = []
    y = []

    for json_file in os_listdir(source_folder):
        with open(f"{source_folder}/{json_file}", "r") as file:
            json_data = json_load(file)

        class_name = json_file.replace("_features.json", "")
        if cfg.experiment.classifier not in ["knn_all", "knn_torch"]:
            if cfg.experiment.use_centroids:
                X.extend(json_data)
            else:
                X.extend(json_data.values())
            y.extend([class_name] * len(json_data))
        else:
            y.extend([class_name])

    if cfg.experiment.classifier not in ["knn_all", "knn_torch"]:
        X = np_array(X)
    y = np_array(y)

    if cfg.experiment.classifier not in ["knn_all", "knn_torch"]:
        classifier.fit(X, y)
    elif cfg.experiment.classifier == "knn_all":
        classifier.fit(source_folder, y)
    else:
        train_folder_for_torch = to_absolute_path(
            f"{embed_folder}/{train_folder}"
        )
        classifier.fit(train_folder_for_torch, y)
    logger.info(f"{cfg.experiment.classifier} classifier fitted.")

    # Save classifier to artifacts folder
    if cfg.experiment.classifier not in ["faiss", "knn_all", "knn_torch"]:
        model_path = f"{target_folder}/model.joblib"
        joblib_dump(classifier, model_path)
    elif cfg.experiment.classifier == "faiss":
        config_path = f"{target_folder}/config.json"
        index_path = f"{target_folder}/index.faiss"
        classifier.save(config_path, index_path)
    else:
        classifier.save(f"{target_folder}/config.json")
    logger.info(f"{cfg.experiment.classifier} classifier saved.")

    # Evaluation
    embed_model_version = cfg.experiment.embed_model_version
    embed_folder = f"data/features/{embed_model_version}"
    eval_folder = cfg.experiment.eval_data_folder
    rel_source_folder = f"{embed_folder}/{eval_folder}/json"
    source_folder = to_absolute_path(rel_source_folder)

    eval_filenames = []
    eval_X = []
    eval_y = []

    for json_file in os_listdir(source_folder):
        with open(f"{source_folder}/{json_file}", "r") as file:
            json_dict = json_load(file)
            json_data = [v for v in json_dict.values()]

        class_name = json_file.replace("_features.json", "")
        if cfg.experiment.classifier != "knn_torch":
            eval_filenames.extend(json_dict.keys())
        else:
            eval_filenames.extend(
                [f"{class_name}/{key}.pt" for key in json_dict.keys()]
            )
        eval_X.extend(json_data)
        eval_y.extend([class_name] * len(json_data))

    if (
        cfg.experiment.classifier != "knn_all"
        and cfg.experiment.classifier != "knn_torch"
    ):
        eval_X = np_array(eval_X)
        eval_y = np_array(eval_y)
        eval_pred_proba = classifier.predict_proba(eval_X)
    elif cfg.experiment.classifier == "knn_all":
        eval_pred_proba = classifier.predict_proba(
            eval_X,
            filenames=eval_filenames,
            folder=to_absolute_path(f"{embed_folder}/{eval_folder}"),
        )
    else:
        eval_pred_proba = classifier.predict_proba(
            filenames=eval_filenames,
            folder=to_absolute_path(f"{embed_folder}/{eval_folder}"),
        )
    eval_pred_proba = [
        handle_zero(truth_class, pred_proba, classifier.classes_.tolist())
        for (truth_class, pred_proba) in zip(eval_y, eval_pred_proba)
    ]

    top_10_acc_scores = []
    for i in range(1, 11):
        logger.info(f"Evaluating top {i} accuracy")
        acc_score = top_k_accuracy_score(
            eval_y, eval_pred_proba, k=i, labels=classifier.classes_
        )
        logger.info(f"Top {i} accuracy: {acc_score}")
        top_10_acc_scores.append(acc_score)

    # Save evaluation results to artifacts folder
    eval_results = {
        "top_10_acc_scores": top_10_acc_scores,
    }
    eval_results_path = f"{target_folder}/eval_results.json"
    with open(eval_results_path, "w") as file:
        json_dump(eval_results, file, indent=4)

    eval_pred_proba_json = {
        k: list(v) for (k, v) in zip(eval_filenames, eval_pred_proba)
    }
    eval_pred_proba_path = f"{target_folder}/eval_pred_proba.json"
    with open(eval_pred_proba_path, "w") as file:
        json_dump(eval_pred_proba_json, file, indent=4)


if __name__ == "__main__":
    # Call this module from the root directory of the project via
    # python -m scripts.run_experiment
    run_experiment()
