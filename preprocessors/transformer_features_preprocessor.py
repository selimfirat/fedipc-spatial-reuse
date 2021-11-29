import torch
from preprocessors.abstract_base_preprocessor import AbstractBasePreprocessor
import math

class TransformerFeaturesPreprocessor(AbstractBasePreprocessor):

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
        labels = torch.zeros((num_data, self.output_size), dtype=torch.float32)

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            threshold_data["threshold"] = [int(threshold_data["threshold"][0])]

            for fname in feature_names:
                new_features = self.scale(fname, torch.FloatTensor(threshold_data[fname]))

                fname = "combined" if fname != "interference" else fname

                if fname == "interference" or fname not in features[idx].keys():
                    features[idx][fname] = new_features
                else:
                    features[idx][fname] = torch.cat([features[idx][fname], new_features], -1)
            num_sta = math.floor(len(features[idx]["combined"]) /2)
            thr = features[idx]["combined"][0]
            rssi = features[idx]["combined"][1:1+num_sta]
            sinr = features[idx]["combined"][1+num_sta:]
            intr = features[idx]["interference"]
            intr_len = len(intr)
            inp = torch.zeros((num_sta,9))
            for i in range(num_sta):
                inp[i][0] = rssi[i]
                inp[i][1] = sinr[i]
                inp[i][2] = thr
                inp[i][3:3+intr_len] = intr
            features[idx]["input"] = inp
            labels[idx, :len(node_labels[threshold])] = torch.FloatTensor(node_labels[threshold])

        return features, labels
