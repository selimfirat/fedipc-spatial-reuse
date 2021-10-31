import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm
from copy import deepcopy

from evaluator import Evaluator
from federated_trainers.abstract_base_federated_trainer import AbstractBaseFederatedTrainer
from utils import to_device


class FedAvgTrainer(AbstractBaseFederatedTrainer):

    def __init__(self, evaluator, logger, **cfg):
        super(FedAvgTrainer, self).__init__(evaluator, logger, **cfg)
        self.patience_left = self.cfg["early_stopping_patience"]

        self.best_mae = float("inf")

        self.best_model = None
        
        self.model = to_device(self.model, self.cfg["device"])

    def train(self, train_loader, val_loader):

        optimizer_state_dicts = {
            context_key: deepcopy(self.optimizer.state_dict()) for context_key, _, _ in train_loader
        }

        for round_idx in tqdm(range(1, self.cfg["max_num_rounds"] + 1)):

            m = max(int(np.round(self.cfg["participation"]*len(train_loader))), 1)
            context_keys = [context_key for context_key, _, _ in train_loader]
            chosen_contexts = [context_keys[i] for i in np.random.choice(list(range(len(train_loader))), m, replace=False)]

            original_state_dict = deepcopy(self.model.state_dict())
            model_state_dicts = {}

            total_loss = .0

            nums_data = {}

            for context_key, context_data_loader, num_data in train_loader:
                if context_key not in chosen_contexts:
                    continue

                self.model.load_state_dict(original_state_dict)
                self.optimizer.load_state_dict(optimizer_state_dicts[context_key])

                total_loss += self.train_node(self.model, self.optimizer, context_data_loader, original_state_dict)

                model_state_dicts[context_key] = deepcopy(self.model.state_dict())
                optimizer_state_dicts[context_key] = deepcopy(self.optimizer.state_dict())

                nums_data[context_key] = num_data

            self.logger.log_metric("train_avg_loss", total_loss / len(chosen_contexts), round_idx)

            self.aggregate(model_state_dicts, chosen_contexts, nums_data)

            if (round_idx % self.cfg["early_stopping_check_rounds"]) == 0:
                eval_val = self.evaluator.calculate(self, val_loader)
                cur_mae = eval_val["mae"]
                self.logger.log_metric("val_mae_es", cur_mae, round_idx)

                if cur_mae < self.best_mae:
                    self.patience_left = self.cfg["early_stopping_patience"]
                    self.best_mae = cur_mae
                    self.best_model = deepcopy(self.model)
                else:
                    self.patience_left -= 1

                if self.patience_left <= 0:
                    self.logger.log_metric("stopped_at_round", round_idx)
                    break
        else:
            eval_val = self.evaluator.calculate(self, val_loader)
            cur_mae = eval_val["mae"]

            if cur_mae < self.best_mae:
                self.best_model = self.model
    
            self.logger.log_metric("stopped_at_round", self.cfg["max_num_rounds"])

        self.model = self.best_model

    def train_node(self, model, optimizer, context_data_loader, original_state_dict):
        model.train()

        total_loss = .0

        for epoch_idx in range(self.cfg["num_epochs"]):

            epoch_total_loss = .0

            # The data are shuffled in preprocessor so no need to shuffle again.
            for X, y, y_len in context_data_loader:
                y_len = y_len[0]
                model.zero_grad()
                optimizer.zero_grad()

                X = to_device(X, self.cfg["device"])
                y_pred = model.forward(X)[:, :y_len]

                y = to_device(y[:, :y_len], self.cfg["device"])

                cur_loss = self.loss(y_pred, y, self.model.state_dict(), original_state_dict)

                cur_loss.backward()

                optimizer.step()

                epoch_total_loss += cur_loss.item() * len(X)

            epoch_avg_loss = epoch_total_loss / len(context_data_loader)
            total_loss += epoch_avg_loss

            avg_loss = total_loss / (epoch_idx + 1)

        return total_loss / self.cfg["num_epochs"]

    def aggregate(self, model_state_dicts, chosen_contexts, nums_data):

        total_data = sum(nums_data.values())
        new_state_dict = {}

        for key in self.model.state_dict().keys():
            new_state_dict[key] = 0.0
            for context_idx in chosen_contexts:
                new_state_dict[key] += model_state_dicts[context_idx][key] * (nums_data[context_idx] / total_data)
                
        self.model.load_state_dict(new_state_dict)

    def predict(self, data_loader):
        self.model.eval()

        y_pred = {}
        y_true = {}
        for context_idx, context_loader, num_data in data_loader:

            for X, y, y_len in context_loader:
                y_len = y_len[0]
                X = to_device(X, self.cfg["device"])

                preds = self.model.forward(X).detach()[:, :y_len].flatten().cpu()
                y = y[:, :y_len].flatten()

                y_pred[context_idx] = torch.cat([y_pred[context_idx], preds], dim=0) if context_idx in y_pred else preds
                y_true[context_idx] = torch.cat([y_true[context_idx], y], dim=0) if context_idx in y_true else y

        return y_true, y_pred
