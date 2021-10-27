import dill as pickle
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SRProcessedDataset
import os


class AbstractBasePreprocessor(ABC):

    def __init__(self, scenario, preprocessor, output_size, batch_size, input_scaler, output_scaler, output_scaler_ins, shuffle, use_cache=True, cache_path="tmp/", **cfg):
        self.use_cache = use_cache
        self.cache_path = os.path.join(cache_path, f"features_{preprocessor}_{input_scaler}_{output_scaler}_scenario{scenario}.pkl")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_size = output_size
        self.output_scaler = output_scaler_ins
        self.scenario = scenario
        self.input_scaler = input_scaler

        self.stats = {
            1: {'throughput': {'min': 0.03, 'max': 110.41, 'mean': 57.818561309523815, 'std': 29.608858485445268}, 'interference': {'min': -151.73, 'max': -0.0, 'mean': -97.49411523611523, 'std': 21.936604553912254}, 'rssi': {'min': -81.41, 'max': -37.42, 'mean': -58.52763214285714, 'std': 7.11873413950484}, 'sinr': {'min': -9.4, 'max': 53.64, 'mean': 32.949347619047614, 'std': 8.618997584824925}, 'threshold': {'min': -82.0, 'max': -62.0, 'mean': -72.0, 'std': 6.05548095703125}},
            2: {'throughput': {'min': 0.0, 'max': 57.88, 'mean': 16.012168194980696, 'std': 11.065256101784948}, 'interference': {'min': -150.16, 'max': -0.0, 'mean': -88.64759573476049, 'std': 26.154485706129986}, 'rssi': {'min': -81.41, 'max': -0.0, 'mean': -58.22506696428571, 'std': 6.869791482037659}, 'sinr': {'min': -20.52, 'max': 55.3, 'mean': 30.648423423423424, 'std': 11.37196171194207}, 'threshold': {'min': -82.0, 'max': -62.0, 'mean': -72.0, 'std': 6.05548095703125}},
            3: {'throughput': {'min': 0.0, 'max': 59.59, 'mean': 16.07963606654783, 'std': 11.011395347524639}, 'interference': {'min': -152.24, 'max': -0.0, 'mean': -88.66472291466923, 'std': 26.151377927825887}, 'rssi': {'min': -81.43, 'max': -36.43, 'mean': -58.24301247771835, 'std': 6.8552820047072025}, 'sinr': {'min': -28.39, 'max': 56.5, 'mean': 30.92333821407351, 'std': 10.94405564797456}, 'threshold': {'min': -82.0, 'max': -62.0, 'mean': -72.0, 'std': 6.0554914474487305}}
        }[scenario] # Obtained via scripts.find_minmax module from the training set.

    def scale(self, feature_name, val):
        if self.input_scaler == "minmax":
            minval = self.stats[feature_name]["min"]
            maxval = self.stats[feature_name]["max"]

            return (val - minval) / (maxval - minval)

        elif self.input_scaler == "standard":
            mean = self.stats[feature_name]["mean"]
            std = self.stats[feature_name]["std"]

            return (val - mean) / std

    @abstractmethod
    def fit(self, train_loader):
        pass

    @abstractmethod
    def transform(self, loader):
        pass

    def _calculate_label_lengths(self, loader):
        res = []

        for context_idx, features, labels in loader:
            r = [len(lbl) for lbl in labels.values()]
            res.append(r)

        return res

    def fit_transform(self, train_data, val_data, test_data):
        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        self.fit(train_data)

        label_lengths = self._calculate_label_lengths(train_data)
        context_indices, features, labels = self.transform(train_data)
        self.output_scaler.fit(labels)
        labels = self.output_scaler.transform(labels)

        new_train_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, label_lengths, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        label_lengths = self._calculate_label_lengths(val_data)
        context_indices, features, labels = self.transform(val_data)
        labels = self.output_scaler.transform(labels)

        new_val_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, label_lengths, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        label_lengths = self._calculate_label_lengths(test_data)
        context_indices, features, labels = self.transform(test_data)
        labels = self.output_scaler.transform(labels)

        new_test_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, label_lengths, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        input_size = features[0].shape[1] if type(features[0]) is torch.Tensor and np.all(np.array([feat.shape[-1] for feat in features]) == features[0].shape[-1]) else "UNKNOWN"
        if input_size == "UNKNOWN" and type(features[0][0]) is dict:
            input_size = f"UNKNOWN_DICT_{len(features[0][0].keys())}"

        res = new_train_loader, new_val_loader, new_test_loader, input_size, self.output_scaler

        if self.use_cache:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        return res


"""
Scenario 1
        throughput {'min': tensor(0.0300, dtype=torch.float64), 'max': tensor(110.4100, dtype=torch.float64), 'mean': tensor(57.8186, dtype=torch.float64), 'std': tensor(29.6089, dtype=torch.float64)}
interference {'min': tensor(-151.7300, dtype=torch.float64), 'max': tensor(-0., dtype=torch.float64), 'mean': tensor(-97.4941, dtype=torch.float64), 'std': tensor(21.9366, dtype=torch.float64)}
rssi {'min': tensor(-81.4100, dtype=torch.float64), 'max': tensor(-37.4200, dtype=torch.float64), 'mean': tensor(-58.5276, dtype=torch.float64), 'std': tensor(7.1187, dtype=torch.float64)}
sinr {'min': tensor(-9.4000, dtype=torch.float64), 'max': tensor(53.6400, dtype=torch.float64), 'mean': tensor(32.9493, dtype=torch.float64), 'std': tensor(8.6190, dtype=torch.float64)}
Scenario 2
throughput {'min': tensor(0., dtype=torch.float64), 'max': tensor(57.8800, dtype=torch.float64), 'mean': tensor(16.0122, dtype=torch.float64), 'std': tensor(11.0653, dtype=torch.float64)}
interference {'min': tensor(-150.1600, dtype=torch.float64), 'max': tensor(-0., dtype=torch.float64), 'mean': tensor(-88.6476, dtype=torch.float64), 'std': tensor(26.1545, dtype=torch.float64)}
rssi {'min': tensor(-81.4100, dtype=torch.float64), 'max': tensor(-0., dtype=torch.float64), 'mean': tensor(-58.2251, dtype=torch.float64), 'std': tensor(6.8698, dtype=torch.float64)}
sinr {'min': tensor(-20.5200, dtype=torch.float64), 'max': tensor(55.3000, dtype=torch.float64), 'mean': tensor(30.6484, dtype=torch.float64), 'std': tensor(11.3720, dtype=torch.float64)}
Scenario 3
throughput {'min': tensor(0., dtype=torch.float64), 'max': tensor(59.5900, dtype=torch.float64), 'mean': tensor(16.0796, dtype=torch.float64), 'std': tensor(11.0114, dtype=torch.float64)}
interference {'min': tensor(-152.2400, dtype=torch.float64), 'max': tensor(-0., dtype=torch.float64), 'mean': tensor(-88.6647, dtype=torch.float64), 'std': tensor(26.1514, dtype=torch.float64)}
rssi {'min': tensor(-81.4300, dtype=torch.float64), 'max': tensor(-36.4300, dtype=torch.float64), 'mean': tensor(-58.2430, dtype=torch.float64), 'std': tensor(6.8553, dtype=torch.float64)}
sinr {'min': tensor(-28.3900, dtype=torch.float64), 'max': tensor(56.5000, dtype=torch.float64), 'mean': tensor(30.9233, dtype=torch.float64), 'std': tensor(10.9441, dtype=torch.float64)}
"""