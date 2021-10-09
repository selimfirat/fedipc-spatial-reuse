from abc import ABC, abstractmethod
from maps import Maps

class AbstractBaseFederatedTrainer(ABC):

    def __init__(self, **cfg):
        self.cfg = cfg
        self.nn_model = self.cfg["nn_model"]
        self.loss = cfg["loss"]

    @abstractmethod
    def train(self, train_loader):
        pass

    @abstractmethod
    def aggregate(self, state_dicts):
        pass

    @abstractmethod
    def predict(self, data_loader):
        pass

    def get_loss(self):

        return Maps.losses[self.loss]()

    def get_nn_model(self):

        return Maps.nn_models[self.nn_model](**self.cfg)
