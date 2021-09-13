import numpy as np
import torch.multiprocessing as mp
import torch
from torch.optim import SGD
from tqdm import tqdm


class FederatedAveragingTrainer:

    def __init__(self, main_model, nn_models, **params):
        self.main_model = main_model
        self.nn_models = nn_models
        self.params = params

        self.optimizers = {context_key: SGD(model.parameters(), lr=self.params["lr"]) for context_key, model in self.nn_models.items()}

    def train(self, nodes_features, nodes_labels, train_contexts):

        for round_idx in tqdm(range(self.params["num_rounds"])):
            m = max(int(self.params["participation"]*len(train_contexts)), 1)
            chosen_nodes = np.random.choice(train_contexts, m, replace=False)

            # Run models in parallel.
            # TODO: add cli param to disable parallel running to save memory etc.
            processes = []
            for context_key in chosen_nodes:
                p = mp.Process(target=self.train_node, args=(self.nn_models[context_key], self.optimizers[context_key], nodes_features[context_key], nodes_labels[context_key]))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            self.aggregate(chosen_nodes)

        return self.nn_models

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

        return self.main_model, self.nn_models

    def aggregate(self, chosen_nodes):
        main_sd = self.main_model.state_dict()
        state_dicts = [self.nn_models[context_key].state_dict() for context_key in chosen_nodes]

        for key in main_sd:
            # TODO: implement n_k / n part (not required for now since all n_k / n values are equal)
            main_sd[key] = torch.mean(torch.stack([sd[key] for sd in state_dicts]), dim=0)

        self.main_model.load_state_dict(main_sd)

        for sd in state_dicts:
            sd.load_state_dict(main_sd)

    def predict_with_main_model(self, nodes_features, target_nodes):

        self.main_model.eval()
        res = torch.empty((len(target_nodes),))
        for idx, context_key in enumerate(target_nodes):
            res[idx] = self.main_model.forward(nodes_features[context_key])

        return res
