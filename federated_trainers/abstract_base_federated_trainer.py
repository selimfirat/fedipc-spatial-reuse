from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, AdamW

from mapper import Mapper


class AbstractBaseFederatedTrainer(ABC):

    def __init__(self, evaluator, logger, nn_model, loss, **cfg):
        self.cfg = cfg
        self.loss_name = loss
        self.logger = logger
        self.evaluator = evaluator

        self.model = Mapper.get_nn_model(nn_model)(**cfg)
        self.optimizer = self.init_optimizer(self.model)

    def init_optimizer(self, model):

        optimizer = None

        if self.cfg["optimizer"] == "sgd":
            optimizer = SGD(model.parameters(), lr=self.cfg["lr"], momentum=self.cfg["momentum"], nesterov=self.cfg["nesterov"],
                dampening=self.cfg["dampening"], weight_decay=self.cfg["weight_decay"])
        elif self.cfg["optimizer"] == "adam":
            optimizer = Adam(model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        elif self.cfg["optimizer"] == "adamw":
            optimizer = AdamW(model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        return optimizer

    def loss(self, y_pred, y, current_state_dict, original_state_dict):
        loss_ins = Mapper.get_loss(self.loss_name)

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
