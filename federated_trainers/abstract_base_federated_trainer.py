from abc import ABC, abstractmethod
from mapper import Mapper


class AbstractBaseFederatedTrainer(ABC):

    def __init__(self, nn_model, loss, **cfg):
        self.model = Mapper.get_nn_model(nn_model)(**cfg)
        self.loss = Mapper.get_loss(loss)()

    @abstractmethod
    def train(self, train_loader):
        pass

    @abstractmethod
    def aggregate(self, state_dicts):
        pass

    @abstractmethod
    def predict(self, data_loader):
        pass
