import torch
from preprocessors.abstract_base_preprocessor import AbstractBasePreprocessor


class SequentialFeaturesPreprocessor(AbstractBasePreprocessor):

    def fit(self, train_loader):

        return self

    def transform(self, loader):

        all_context_indices, all_features, all_labels = [], [], []

        for context_idx, features, labels in loader:
            features, labels = self._process_node(features, labels)

            all_context_indices.append(context_idx)
            all_features.append(features)
            all_labels.append(labels)

        return all_context_indices, all_features, all_labels

    def _process_node(self, node_data, node_labels):

        feature_names = ["threshold", "interference", "rssi", "sinr"]

        num_data = len(node_data.keys())

        features = [{} for _ in range(num_data)]
        labels = torch.empty((num_data,))

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            threshold_data["threshold"] = int(threshold_data["threshold"])

            for fname in feature_names:
                features[idx][fname] = torch.FloatTensor(threshold_data[fname])

            labels[idx] = node_labels[threshold]

        return features, labels
