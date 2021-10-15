import torch

from federated_trainers.fed_avg_trainer import FedAvgTrainer


class FedProxTrainer(FedAvgTrainer):

    def __init__(self, **cfg):
        super(FedProxTrainer, self).__init__(**cfg)

        old_loss = self.loss

        proximal_term = lambda current_state_dict, original_state_dict: 0.5 * self.cfg["mu"] * torch.sum(torch.stack([torch.sum((current_state_dict[key] - original_state_dict[key])**2) for key in current_state_dict.keys()]))

        self.loss = lambda y_pred, y, current_state_dict, original_state_dict: old_loss(y_pred, y, current_state_dict, original_state_dict) + proximal_term(current_state_dict, original_state_dict)
