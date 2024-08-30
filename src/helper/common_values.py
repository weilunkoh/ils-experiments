# Enum for embedding model names
EMBED_MODEL_NAMES = ["clip", "resnet50", "inceptionV3"]

# Enum for clustering methods
CLUSTERING_METHODS = ["kmeans", "minibatchkmeans"]

# Enum for centroid intialisation methods
CENTROID_INIT_METHODS = ["k-means++", "incremental"]

# Enum for classifier methods that use centroids
CENTROID_CLASSIFIER_METHODS = ["knn", "logreg", "faiss"]

# Enum for classifier methods that use image features directly
CLASSIFIER_METHODS = ["faiss", "knn_all", "knn_torch"]
