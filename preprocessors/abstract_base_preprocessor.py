import dill as pickle
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SRProcessedDataset
import os


class AbstractBasePreprocessor(ABC):

    def __init__(self, scenario, preprocessor, batch_size, input_normalizer_instance, shuffle, use_cache=True, cache_path="tmp/", **cfg):
        self.use_cache = use_cache
        self.cache_path = os.path.join(cache_path, f"features_{preprocessor}_scenario{scenario}.pkl")
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.input_normalizer = input_normalizer_instance

    @abstractmethod
    def fit(self, train_loader):
        pass

    @abstractmethod
    def transform(self, loader):
        pass

    def fit_transform(self, train_loader, test_loader):
        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        self.fit(train_loader)

        context_indices, features, labels = self.transform(train_loader)
        self.input_normalizer.fit(features)
        features = self.input_normalizer.transform(features)

        new_train_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

        context_indices, features, labels = self.transform(test_loader)
        features = self.input_normalizer.transform(features)

        new_test_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)


        input_size = features[0].shape[1] if type(features[0]) is torch.Tensor and np.all(np.array([feat.shape[-1] for feat in features]) == features[0].shape[-1]) else "UNKNOWN"
        if input_size == "UNKNOWN" and type(features[0][0]) is dict:
            input_size = f"UNKNOWN_DICT_{len(features[0][0].keys())}"

        if self.use_cache:
            with open(self.cache_path, 'wb') as f:
                pickle.dump((new_train_loader, new_test_loader, input_size), f, protocol=pickle.HIGHEST_PROTOCOL)

        return new_train_loader, new_test_loader, input_size
