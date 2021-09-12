import numpy as np
import os
import pickle
from sklearn.utils import shuffle

class Preprocessor:

    def __init__(self, preprocess_type="basic_features", use_cache=True, cache_path="tmp/", shuffle_per_node=True):
        self.preprocess_type = preprocess_type
        self.shuffle_per_node = shuffle_per_node
        self.use_cache = use_cache
        self.cache_path = os.path.join(cache_path, f"features_{preprocess_type}.pkl")

    def apply_noncached(self, nodes_data, y_true_dict):
        func = getattr(self, self.preprocess_type)

        features = {sim_no: {} for sim_no in nodes_data.keys()}
        labels = {}

        for sim_no, node in nodes_data.items():
            features[sim_no], labels[sim_no] = func(node, y_true_dict[sim_no])

        return features, labels

    def apply(self, nodes_data, y_true_dict):
        if not self.use_cache:
            return self.apply_noncached(nodes_data, y_true_dict)

        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        # Load input files
        features, labels = self.apply_noncached(nodes_data, y_true_dict)

        if self.shuffle_per_node:
            features, labels = shuffle(features, labels, random_state=1)

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump((features, labels), f, protocol=pickle.HIGHEST_PROTOCOL)

        return features, labels

    def basic_features(self, node_data, node_labels):

        feature_names = ["threshold", "interference", "rssi", "sinr"]

        num_data = len(node_data.keys())
        num_features = len(feature_names)

        features = np.empty((num_data, num_features))
        labels = np.empty((num_data, ))

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            threshold_data["threshold"] = int(threshold_data["threshold"])

            for fi in range(len(feature_names)):
                features[idx, fi] = np.mean(threshold_data[feature_names[fi]])

            labels[idx] = node_labels[threshold]

        return features, labels
