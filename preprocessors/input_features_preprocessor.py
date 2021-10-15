import torch
from preprocessors.abstract_base_preprocessor import AbstractBasePreprocessor

class InputFeaturesPreprocessor(AbstractBasePreprocessor):

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

        feature_names = "x(m);y(m);z(m);central_freq(GHz);channel_bonding_model;primary_channel;min_channel_allowed;max_channel_allowed;tpc_default(dBm);cca_default(dBm);traffic_model;traffic_load[pkt/s];packet_length;num_packets_aggregated;capture_effect_model;capture_effect_thr;constant_per;pifs_activated;cw_adaptation;cont_wind;cont_wind_stage;bss_color;spatial_reuse_group;non_srg_obss_pd;srg_obss_pd".split(";")

        num_data = len(node_data.keys())
        num_features = len(feature_names) * (2 if self.scenario == 1 else 5)

        features = torch.zeros((num_data, num_features), dtype=torch.float32)
        labels = torch.empty((num_data,), dtype=torch.float32)

        node_codes = ["AP_A", "STA_A1", "STA_A2", "STA_A3", "STA_A4"]

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            labels[idx] = node_labels[threshold]

            feat_idx = 0

            input_data = threshold_data["input_nodes"]
            for node_idx, node_code in input_data["node_code"].items():
                if node_code[0] in node_codes:
                    for feature_name, val in input_data.items():
                        if feature_name in feature_names:
                            features[idx][feat_idx] = val[node_idx][0].float()
                            feat_idx += 1

        return features, labels
