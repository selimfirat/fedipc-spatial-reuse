import torch

from federated_trainers.fed_avg_trainer import FedAvgTrainer


class FedProxTrainer(FedAvgTrainer):

    def __init__(self, **cfg):
        super(FedProxTrainer, self).__init__(**cfg)

    def proximal_term(self, current_state_dict, original_state_dict):

        return 0.5 * self.cfg["mu"] * torch.sum(torch.stack([torch.sum((current_state_dict[key] - original_state_dict[key])**2) for key in current_state_dict.keys()]))

    def loss(self, y_pred, y, current_state_dict, original_state_dict):

        #print(self.proximal_term(current_state_dict, original_state_dict))

        return super().loss(y_pred, y, current_state_dict, original_state_dict) + self.proximal_term(current_state_dict, original_state_dict)
