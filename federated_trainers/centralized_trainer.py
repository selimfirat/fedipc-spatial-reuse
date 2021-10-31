import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm
from copy import deepcopy

from evaluator import Evaluator
from federated_trainers.abstract_base_federated_trainer import AbstractBaseFederatedTrainer
from utils import to_device


class CentralizedTrainer(AbstractBaseFederatedTrainer):

    def __init__(self, evaluator, logger, nn_model, loss, **cfg):
        super(CentralizedTrainer, self).__init__(evaluator, logger, nn_model, loss, **cfg)
        self.cfg = cfg

        self.patience_left = self.cfg["early_stopping_patience"]

        self.best_mae = float("inf")

        self.best_model = None

    def train(self, train_loader, val_loader):

        for round_idx in tqdm(range(1, self.cfg["max_num_rounds"] + 1)):

            m = max(int(np.round(self.cfg["participation"]*len(train_loader))), 1)
            chosen_contexts = np.random.choice(list(range(len(train_loader))), m, replace=False)

            optimizer = SGD(self.model.parameters(), lr=self.cfg["lr"], momentum=self.cfg["momentum"],
                            nesterov=self.cfg["nesterov"], dampening=self.cfg["dampening"],
                            weight_decay=self.cfg["weight_decay"])
            total_loss = .0
            self.model = to_device(self.model, self.cfg["device"])

            self.model.train()

            for epoch_idx in range(self.cfg["num_epochs"]):

                for i, (context_key, context_data_loader, num_data) in enumerate(train_loader):

                    for X, y in context_data_loader:
                        self.model.zero_grad()
                        optimizer.zero_grad()

                        X = to_device(X, self.cfg["device"])
                        y_pred = self.model.forward(X)

                        y = to_device(y, self.cfg["device"])

                        cur_loss = self.loss(y_pred, y, self.model.state_dict(), {})

                        cur_loss.backward()

                        optimizer.step()

                        total_loss += cur_loss.item() * len(X)

                        avg_loss = total_loss / (epoch_idx + 1)

            self.model = to_device(self.model, "cpu")

            self.logger.log_metric("train_avg_loss", total_loss / len(chosen_contexts), round_idx)

            if (round_idx % self.cfg["early_stopping_check_rounds"]) == 0:
                eval_val = self.evaluator.calculate(self, val_loader)
                cur_mae = eval_val["mae"]
                self.logger.log_metric("val_mae_es", cur_mae, round_idx)

                if cur_mae < self.best_mae and eval_val["r2"] > 0:
                    self.patience_left = self.cfg["early_stopping_patience"]
                    self.best_mae = cur_mae
                    self.best_model = deepcopy(self.model)
                else:
                    self.patience_left -= - 1

                if self.patience_left <= 0:
                    self.logger.log_metric("stopped_at_round", round_idx)
                    break
        else:
            self.best_model = self.model
            self.logger.log_metric("stopped_at_round", self.cfg["max_num_rounds"])

        self.model = self.best_model

    def train_node(self, model, optimizer, context_data_loader, original_state_dict):
        pass

    def aggregate(self, state_dicts):
        pass

    def predict(self, data_loader):
        self.model.eval()

        y_pred = {}
        y_true = {}
        for context_idx, context_loader, num_data in data_loader:

            for X, y in context_loader:

                preds = self.model.forward(X).detach()

                y_pred[context_idx] = torch.cat([y_pred[context_idx], preds], dim=0) if context_idx in y_pred else preds

                y_true[context_idx] = torch.cat([y_true[context_idx], y], dim=0) if context_idx in y_true else y

        return y_true, y_pred
