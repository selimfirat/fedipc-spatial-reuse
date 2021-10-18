from abc import ABC, abstractmethod
from mapper import Mapper


class AbstractBaseFederatedTrainer(ABC):

    def __init__(self, logger, nn_model, loss, **cfg):
        self.model = Mapper.get_nn_model(nn_model)(**cfg)

        self.loss_name = loss
        self.logger = logger

    def loss(self, y_pred, y, current_state_dict, original_state_dict):
        loss_ins = Mapper.get_loss(self.loss_name)()

        return loss_ins(y_pred, y)

    @abstractmethod
    def train_node(self, model, optimizer, context_data_loader, original_state_dict):
        pass

    @abstractmethod
    def aggregate(self, state_dicts):
        pass

    @abstractmethod
    def predict(self, data_loader):
        pass
