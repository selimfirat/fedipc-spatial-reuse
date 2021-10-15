import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm
from copy import deepcopy
from federated_trainers.abstract_base_federated_trainer import AbstractBaseFederatedTrainer
from utils import to_device


class FedAvgTrainer(AbstractBaseFederatedTrainer):

    def __init__(self, **cfg):
        super(FedAvgTrainer, self).__init__(**cfg)

        self.cfg = cfg

    def train(self, train_loader):

        for round_idx in tqdm(range(self.cfg["num_rounds"])):
            m = max(int(np.round(self.cfg["participation"]*len(train_loader))), 1)
            chosen_contexts = np.random.choice(list(range(len(train_loader))), m, replace=False)

            original_state_dict = deepcopy(self.model.state_dict())
            state_dicts = {}

            for i, (context_key, context_data_loader) in enumerate(train_loader):
                if i not in chosen_contexts:
                    continue

                self.model.load_state_dict(original_state_dict)
                self.model = to_device(self.model, self.cfg["device"])
                optimizer = SGD(self.model.parameters(), lr=self.cfg["lr"])

                self.train_node(self.model, optimizer, context_data_loader, original_state_dict)
                self.model = to_device(self.model, "cpu")

                state_dicts[context_key] = deepcopy(self.model.state_dict())

            self.aggregate(state_dicts)

    def train_node(self, model, optimizer, context_data_loader, original_state_dict):
        model.train()

        total_loss = .0

        for epoch_idx in range(self.cfg["num_epochs"]):

            epoch_total_loss = .0

            # The data are shuffled in preprocessor so no need to shuffle again.
            for X, y in context_data_loader:
                model.zero_grad()
                optimizer.zero_grad()

                X = to_device(X, self.cfg["device"])
                y_pred = model.forward(X)[:, 0]

                y = to_device(y, self.cfg["device"])
                cur_loss = self.loss(y_pred, y, self.model.state_dict(), original_state_dict)

                cur_loss.backward()

                optimizer.step()

                epoch_total_loss += cur_loss.item() * len(X)

            epoch_avg_loss = epoch_total_loss / len(context_data_loader)
            total_loss += epoch_avg_loss
            avg_loss = total_loss / (epoch_idx + 1)

    def aggregate(self, state_dicts):

        new_state_dict = {}

        for key in self.model.state_dict().keys():
            # TODO: implement n_k / n part (not required for now since all n_k / n values are equal)
            new_state_dict[key] = torch.mean(torch.stack([sd[key].float() for sd in state_dicts.values()], dim=0), dim=0)

        self.model.load_state_dict(new_state_dict)

    def predict(self, data_loader):
        self.model.eval()

        y_pred = {}
        y_true = {}
        for context_idx, context_loader in data_loader:

            for X, y in context_loader:

                preds = self.model.forward(X).detach()

                y_pred[context_idx] = torch.cat([y_pred[context_idx], preds], dim=0) if context_idx in y_pred else preds

                y_true[context_idx] = torch.cat([y_true[context_idx], y], dim=0) if context_idx in y_true else y

        return y_true, y_pred
