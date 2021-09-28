import numpy as np
import os
import torch
import pickle
from sklearn.utils import shuffle


class Preprocessor:

    def __init__(self, preprocess_type="basic_features", scenario=1, use_cache=True, cache_path="tmp/", shuffle_per_node=True):
        self.preprocess_type = preprocess_type
        self.scenario = scenario
        self.shuffle_per_node = shuffle_per_node
        self.use_cache = use_cache

        self.cache_path = os.path.join(cache_path, f"features_{self.preprocess_type}_scenario{self.scenario}.pkl")

    def apply_noncached(self, nodes_data, y_true_dict, train_contexts, test_contexts):
        func = getattr(self, self.preprocess_type)

        return func(nodes_data, y_true_dict, train_contexts, test_contexts)

    def apply(self, nodes_data, y_true_dict, train_contexts, test_contexts):
        if not self.use_cache:
            return self.apply_noncached(nodes_data, y_true_dict, train_contexts, test_contexts)

        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        # Load input files
        features, labels, thresholds = self.apply_noncached(nodes_data, y_true_dict, train_contexts, test_contexts)

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump((features, labels, thresholds), f, protocol=pickle.HIGHEST_PROTOCOL)

        return features, labels, thresholds

    def basic_features(self, nodes_data, y_true_dict, train_contexts, test_contexts):

        def _basic_features_process_node(node_data, node_labels, node_type):

            feature_names = ["threshold", "interference", "rssi", "sinr"]

            num_data = len(node_data.keys())
            num_features = len(feature_names)

            features = torch.empty((num_data, num_features))
            labels = torch.empty((num_data,))
            thresholds = list(node_data.keys())

            for idx, (threshold, threshold_data) in enumerate(node_data.items()):

                threshold_data["threshold"] = int(threshold_data["threshold"])

                for fi in range(len(feature_names)):
                    features[idx, fi] = torch.from_numpy(np.array(np.mean(threshold_data[feature_names[fi]])))

                labels[idx] = node_labels[threshold]

            return features, labels, thresholds

        features = {sim_no: {} for sim_no in nodes_data.keys()}
        labels = {}
        thresholds = {}

        for sim_no, node in nodes_data.items():
            node_type = "train" if sim_no in train_contexts else "test"

            features[sim_no], labels[sim_no], thresholds[sim_no] = _basic_features_process_node(node, y_true_dict[sim_no], node_type)

            if self.shuffle_per_node:
                features[sim_no], labels[sim_no], thresholds[sim_no] = shuffle(features[sim_no], labels[sim_no], thresholds[sim_no], random_state=1)

        return features, labels, thresholds

