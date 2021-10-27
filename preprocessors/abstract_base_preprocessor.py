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

    def fit_transform(self, train_loader, val_loader, test_loader):
        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        self.fit(train_loader)

        label_lengths = self._calculate_label_lengths(train_loader)
        context_indices, features, labels = self.transform(train_loader)
        self.output_scaler.fit(labels)
        labels = self.output_scaler.transform(labels)

        new_train_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, label_lengths, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        label_lengths = self._calculate_label_lengths(val_loader)
        context_indices, features, labels = self.transform(val_loader)
        labels = self.output_scaler.transform(labels)

        new_val_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, label_lengths, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        label_lengths = self._calculate_label_lengths(test_loader)
        context_indices, features, labels = self.transform(test_loader)
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