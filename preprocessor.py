import numpy as np


class Preprocessor:

    def __init__(self, preprocess_type="basic_features"):
        self.preprocess_type = preprocess_type

    def apply(self, nodes_data, y_true_dict):
        func = getattr(self, self.preprocess_type)

        features = {sim_no: {} for sim_no in nodes_data.keys()}
        labels = {}

        for sim_no, node in nodes_data.items():
            features[sim_no], labels[sim_no] = func(node, y_true_dict[sim_no])

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
