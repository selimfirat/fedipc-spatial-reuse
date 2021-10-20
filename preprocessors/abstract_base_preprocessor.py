import dill as pickle
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SRProcessedDataset
import os


class AbstractBasePreprocessor(ABC):

    def __init__(self, scenario, preprocessor, output_size, batch_size, input_scaler, output_scaler, input_scaler_ins, output_scaler_ins, shuffle, use_cache=True, cache_path="tmp/", **cfg):
        self.use_cache = use_cache
        self.cache_path = os.path.join(cache_path, f"features_{preprocessor}_{input_scaler}_{output_scaler}_scenario{scenario}.pkl")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_size = output_size
        self.input_scaler = input_scaler_ins
        self.output_scaler = output_scaler_ins
        self.scenario = scenario

    @abstractmethod
    def fit(self, train_loader):
        pass

    @abstractmethod
    def transform(self, loader):
        pass

    def fit_transform(self, train_loader, val_loader, test_loader):
        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        self.fit(train_loader)

        context_indices, features, labels = self.transform(train_loader)
        self.input_scaler.fit(features)
        features = self.input_scaler.transform(features)

        self.output_scaler.fit(labels)
        labels = self.output_scaler.transform(labels)

        new_train_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        context_indices, features, labels = self.transform(val_loader)
        features = self.input_scaler.transform(features)
        labels = self.output_scaler.transform(labels)

        new_val_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        context_indices, features, labels = self.transform(test_loader)
        features = self.input_scaler.transform(features)
        labels = self.output_scaler.transform(labels)

        new_test_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        input_size = features[0].shape[1] if type(features[0]) is torch.Tensor and np.all(np.array([feat.shape[-1] for feat in features]) == features[0].shape[-1]) else "UNKNOWN"
        if input_size == "UNKNOWN" and type(features[0][0]) is dict:
            input_size = f"UNKNOWN_DICT_{len(features[0][0].keys())}"

        res = new_train_loader, new_val_loader, new_test_loader, input_size, self.input_scaler, self.output_scaler

        if self.use_cache:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        return res
