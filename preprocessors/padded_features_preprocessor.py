import torch
from preprocessors.abstract_base_preprocessor import AbstractBasePreprocessor


class PaddedFeaturesPreprocessor(AbstractBasePreprocessor):

    def fit(self, train_loader):

        return self

    def transform(self, loader):
        all_context_indices, all_features, all_labels = [], [], []

        feature_names = ["threshold", "interference", "rssi", "sinr"]
        max_lens = {}
        for fname in feature_names:
            max_len = -1

            for context_idx, features, labels in loader:
                for idx, (threshold, threshold_data) in enumerate(features.items()):

                    max_len = max(max_len, len(threshold_data[fname]))

            max_lens[fname] = max_len

        for context_idx, features, labels in loader:

            features, labels = self._process_node(features, labels, max_lens)

            all_context_indices.append(context_idx)
            all_features.append(features)
            all_labels.append(labels)

        return all_context_indices, all_features, all_labels

    def _process_node(self, node_data, node_labels, max_lens):

        num_data = len(node_data.keys())
        labels = torch.zeros((num_data, self.output_size), dtype=torch.float32)

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):
            threshold_data["threshold"] = [int(threshold_data["threshold"][0])]

            labels[idx, :len(node_labels[threshold])] = torch.FloatTensor(node_labels[threshold])

        feature_names = ["threshold", "interference", "rssi", "sinr"]
        all_features = []
        for fname in feature_names:

            features = torch.zeros((num_data, max_lens[fname]), dtype=torch.float32)

            for idx, (threshold, threshold_data) in enumerate(node_data.items()):
                cur_features = torch.FloatTensor(threshold_data[fname])
                cur_features = self.scale(fname, cur_features)
                features[idx, :cur_features.shape[0]] = cur_features

            all_features.append(features)

        all_features = torch.cat(all_features, dim=1)

        return all_features, labels
