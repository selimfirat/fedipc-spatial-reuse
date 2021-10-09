from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from dataset import ContextDataset, SRProcessedDataset


class AbstractBasePreprocessor(ABC):

    def __init__(self, batch_size, shuffle, **cfg):
        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def fit(self, train_loader):
        pass

    @abstractmethod
    def transform(self, loader):
        pass


    def fit_transform(self, train_loader, test_loader):

        self.fit(train_loader)

        context_indices, features, labels = self.transform(train_loader)
        new_train_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0])

        context_indices, features, labels = self.transform(test_loader)
        new_test_loader = DataLoader(SRProcessedDataset(context_indices, features, labels, self.batch_size, self.shuffle), collate_fn=lambda x: x[0])

        return new_train_loader, new_test_loader
