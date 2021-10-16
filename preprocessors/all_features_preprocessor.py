import torch
from preprocessors.abstract_base_preprocessor import AbstractBasePreprocessor


class AllFeaturesPreprocessor(AbstractBasePreprocessor):

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
        num_features = len(feature_names)

        features = torch.empty((num_data, num_features), dtype=torch.float32)
        labels = torch.empty((num_data,), dtype=torch.float32)

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            threshold_data["threshold"] = [int(threshold_data["threshold"][0])]

            for fi in range(len(feature_names)):
                features[idx, fi] = torch.mean(torch.FloatTensor(threshold_data[feature_names[fi]]))

            labels[idx] = node_labels[threshold]


        input_features, _ = self._process_node_input_features(node_data, node_labels)

        features = torch.cat([features, input_features], dim=1)

        return features, labels


    def _process_node_input_features(self, node_data, node_labels):

        #feature_names = "x(m);y(m);z(m);central_freq(GHz);channel_bonding_model;primary_channel;min_channel_allowed;max_channel_allowed;tpc_default(dBm);cca_default(dBm);traffic_model;traffic_load[pkt/s];packet_length;num_packets_aggregated;capture_effect_model;capture_effect_thr;constant_per;pifs_activated;cw_adaptation;cont_wind;cont_wind_stage;bss_color;spatial_reuse_group".split(";")
        feature_names = "non_srg_obss_pd;x(m);y(m)".split(";") # changing features only # removed non_srg_obss_pd;x(m);y(m)


        num_data = len(node_data.keys())
        num_features =  (1 if self.scenario == 1 else 4) + len(feature_names) * (2 if self.scenario == 1 else 5) # 1 for AP and 4 for each "STA per AP". Only includes STAs connected to AP_A

        features = torch.zeros((num_data, num_features), dtype=torch.float32)
        labels = torch.empty((num_data,), dtype=torch.float32)

        node_codes = ["AP_A", "STA_A1", "STA_A2", "STA_A3", "STA_A4"]

        for idx, (threshold, threshold_data) in enumerate(node_data.items()):

            labels[idx] = node_labels[threshold]

            feat_idx = 0

            ap_a_loc = (None, None)

            dists = []

            input_data = threshold_data["input_nodes"]
            for node_idx, node_code in input_data["node_code"].items():
                if node_code[0] in node_codes:
                    for feature_name in feature_names:
                        features[idx][feat_idx] = input_data[feature_name][node_idx][0].float()
                        feat_idx += 1

                if node_code[0] == "AP_A":
                    ap_a_loc = (input_data["x(m)"][node_idx][0].float(), input_data["y(m)"][node_idx][0].float())
                if "STA_A" in node_code[0]:
                    dist = torch.sqrt((input_data["x(m)"][node_idx][0].float() - ap_a_loc[0])**2 + (input_data["y(m)"][node_idx][0].float() - ap_a_loc[1])**2)

                    dists.append(dist)

            for dist in dists:
                features[idx][feat_idx] = dist
                feat_idx += 1

        return features, labels
