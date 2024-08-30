# ils-experiments (Interactive Learning with Safeguards)
This is a code repository for conducting experiments to enable:
- Interactive Learning:
    - Choosing models that facilitates collaboration among researchers when training machine learning
- Safeguards:
    - Ensuring images uploaded by researchers are robust enough to mistakes, maintaining integrity of the chosen model as a result

Paper Publication: https://doi.org/10.1016/j.mlwa.2024.100583

## Pre-requisites
- **Python Environment**

  Create a new `Python 3.10` environment via [Conda](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) by runing the following command that creates an environment and installs Python packages.
  ```bash
  conda env create -f ./conda-env.yml
  # or if GPUs are available
  conda env create -f ./conda-env-gpu.yml
  ```

  If there are subsequent changes to the pip requirements, the following command can be used to update the environment.
  ```bash
  conda env update -f ./conda-env.yml
  # or if GPUs are available
  conda env update -f ./conda-env-gpu.yml
  ```

- Access to SMU's Crimson GPU Cluster (optional)

  If there is access, the `.sh` scripts catered specifically for the cluster can be used. Otherwise, the `.py` files in the `scripts` folder should be used instead.

## Initialisation Scripts

- `00_check_torch_gpu.sh`
    - This is simple script to check whether PyTorch is able to access GPUs in the SMU crimson cluster environment.
- `01_download_images.sh` / `scripts/download_images.py`
    - This is a script for downloading images for ImageNet 2012 dataset. For the train set and validation set, this script also produces a class mapping `.json` file that maps each class ID to a set of class names. Both class mapping `.json` files should have the same content.
    - `data/modified_class_mapping.json` is manually edited from one of the class mapping `.json` files such that the first item of each class ID's list of class names is not duplicated with the other classes. The list of changes are:
        - n02012849's "crane" is changed to "crane bird"
        - n03126707's "crane" is changed to "crane machine"
        - n03710721 has "mailot" removed to avoid duplicate with n03710637, resulting in n03710721 having only "tank suit" left in its list

## Experiments for Interactive Learning

### Scripts Design
The scripts in this section are designed to be connected such that each script will check if prerequisites are fulfilled by prior scripts to avoid duplicated work. If prior scripts are not fulfilled, the script is able to execute those prior scripts.

For example, in the `compute_centroids` script, the script will first check if the specified features (e.g. `CLIP` or `ResNet`) are extracted and saved into files in the `data` folder. If the files exist in the `data` folder, `compute_centroids` will proceed with computing centroids (e.g. via `k-means`) without re-extracting the features specified. If the files do not exist, `compute_centroids` will trigger the prior script `extract_features` before proceeding with centroids computation.

### Hydra Yaml Configurations
The various experiment variables are defined in `config/experiment.yaml` file and processed by the Python `hydra` package. The `config/experiment.yaml` file lists examples of values that have code logic catered for. The examples are in comments format in the `.yaml` file.

Outputs of experiments are saved in the `outputs` folder based on date and time. Experiments that have the results shown in `notebooks/results_visualisation.ipynb` are stored in the `outputs/from_server` folder. More on experiment results can be found in the [Results for Experiments](##Results-for-Experiments) section.

### Description of Scripts

- `02_extract_features.sh` / `scripts/extract_features.py`
    - This script is used to extract features (i.e. embedding vectors) from the images based on the specified image embedding model.
- `03_compute_centroids.sh` / `scripts/compute_centroids.py`
    - Using the specified clustering algorithm, this script is used to compute centroids for each class based on the respective set of embeddings extracted earlier.
- `04_run_experiment.sh` / `scripts/run_experiment.py`
    - This file is used to fit the specified classifier based on the centroids computed earlier for each class.

## Experiments for Robustness Safeguards

### Scripts Design
Similar to the above experiments for interactive learning, the scripts in this section are designed to be connected such that each script will check if prerequisites are fulfilled by prior scripts to avoid duplicated work. If prior scripts are not fulfilled, the script is able to execute those prior scripts.

### Description of Scripts
- `05_extract_class_text_features.sh` / `scripts/extract_class_text_features.py`
    - This script is to extract text features for the first class name of each class ID in `data/modified_class_mapping.json`. The text features of each class name are saved to `data/train_class_text_features.json`.
- `06_compute_robustness_thresholds.sh` / `scripts/compute_robustness_thresholds.py`
    - This script calculates the cosine similarity scores of the `train` images with their respective class names. The scores are saved in `.json` files named after their class ID in the `data/robustness_check/train` folder. The lowest cosine similarity score of each class are consolidated and stored in `data/train_robustness_thresholds.json`.
- `07_val_robustness_check.sh` / `scripts/val_robustness_check.py`
    - This script calculates the cosine similarity scores of the `validation` images with their respective class names. The scores are saved in `.json` files named after their class ID in the `data/robustness_check/val` folder. The  scores from `data/train_robustness_thresholds.json` are used as the threshold to calculate the metrics for each class. The negative class for each class are the images from all other classes, resulting in a highly imbalanced dataset.

## Results for Experiments
- `results_visualisation.ipynb`
    - This notebook is used to plot charts to visualise the results of each experiment and to produce a consolidated view of the results of all experiments in `results.xlsx`.
    - The eventual model used for AWS deployment is based on `CLIP` image embeddings, `k-means` with 30 centroids and `knn` classifier with 30 neighbours.
- `robustness_analysis.ipynb`
    - This notebook explores the best threshold to implement the robustness checks.
        - The first hypothesis was to use lowest cosine similarity score found in `data/train_robustness_thresholds.json` when running `06_compute_robustness_thresholds.sh` / `scripts/compute_robustness_thresholds.py`. However, this proved to be unsuitable because of the high number of false positives.
        - The second hypothesis was to show that a suitable threshold exists and this was proven by showing how number of false positives can be drastically reduced when the threshold was increased to not use the lowest cosine similarity score.
        - The third hypothesis was to show that a threshold can be predetermined and this was proven by showing that the 5th percentile threshold roughly leads to similar true negative rates and true positive rates at around 95%.
    - Based on the 5th percentile threshold derived from the third hypothesis, the chosen percentile to use for deployment is eventually set at 15% to prevent too many false alarms.

## Transiting to AWS Deployment
- `scripts/change_model_classes.py`
    - Once the satisfied model is trained in the server, it can be downloaded to a local machine for the final processing step before deploying the model to AWS via the [AWS ILS](https://github.com/weilunkoh/aws-ils) repository. The final processing step is to update the classifier model to use class names instead of class IDs for the classes to enable robustness checks for subsequent updates to the model.

## Update 04/08/2024
- Comparison with a related work by Nakata is added to the list of experiments. (https://arxiv.org/abs/2204.01186v2)
- The paper by Nakata used KNN classification directly on CLIP embeddings while previous 5 experiments in this code repository have used centroids as class representatives.
- Results can be seen under `Experiment 6` in the `results_visualisation.ipynb` notebook.
