import numpy as np
import torch.multiprocessing as mp
import torch
from torch.optim import SGD
from tqdm import tqdm
from federated_trainers.abstract_base_federated_trainer import AbstractBaseFederatedTrainer
from copy import deepcopy

class FederatedAveragingTrainer(AbstractBaseFederatedTrainer):

    def __init__(self, model, **params):
        self.model = model
        self.params = params

        self.optimizer = SGD(model.parameters(), lr=self.params["lr"])



    def train(self, nodes_features, nodes_labels, train_contexts):

        for round_idx in tqdm(range(self.params["num_rounds"])):
            m = max(int(self.params["participation"]*len(train_contexts)), 1)
            chosen_nodes = np.random.choice(train_contexts, m, replace=False)

            # Run models in parallel.
            original_state_dict = deepcopy(self.model.state_dict())
            state_dicts = {}
            for context_key in chosen_nodes:
                self.model.load_state_dict(original_state_dict)
                self.train_node(self.model, self.optimizer, nodes_features[context_key], nodes_labels[context_key])
                state_dicts[context_key] = deepcopy(self.model.state_dict())

            self.aggregate(state_dicts)

    def train_node(self, model, optimizer, X, y):
        batch_size = self.params["batch_size"]

        for epoch_idx in range(self.params["num_epochs"]):
            model.train()

            total_loss = .0

            # The data are shuffled in preprocessor so no need to shuffle again.
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]

                y_batch_pred = model.forward(X_batch)

                cur_loss = ((y_batch - y_batch_pred)**2).mean()

                cur_loss.backward()

                optimizer.step()

                total_loss += cur_loss.item()

            avg_loss = total_loss / (X.shape[0] // batch_size + (1 if X.shape[0] % batch_size == 0 else 0))

    def aggregate(self, state_dicts):

        new_state_dict = {}

        cur_state_dict = self.model.state_dict()

        for key in cur_state_dict.keys():
            # TODO: implement n_k / n part (not required for now since all n_k / n values are equal)
            new_state_dict[key] = torch.mean(torch.stack([sd[key] for sd in state_dicts.values()]), dim=0)

        self.model.load_state_dict(new_state_dict)

    def predict(self, nodes_features, target_nodes):

        self.model.eval()
        res = torch.empty((len(target_nodes),))
        for idx, context_key in enumerate(target_nodes):
            res[idx] = self.model.forward(nodes_features[context_key])

        return res
