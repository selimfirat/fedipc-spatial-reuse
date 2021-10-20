import torch

from federated_trainers.fed_avg_trainer import FedAvgTrainer


class FedProxTrainer(FedAvgTrainer):

    def __init__(self, evaluator, logger, **cfg):
        super(FedProxTrainer, self).__init__(evaluator, logger, **cfg)

    def proximal_term(self, current_state_dict, original_state_dict):
        diff_sum = 0.0
        for key in original_state_dict.keys():
            modules = key.split(".")
            if len(modules) == 3:
                diff_sum += torch.sum(( getattr(getattr(self.model, modules[0])[int(modules[1])], modules[2]) )**2)
            elif len(modules) == 2:
                diff_sum += torch.sum(( getattr(getattr(self.model, modules[0]), modules[1]) )**2)

        return 0.5 * self.cfg["mu"] * diff_sum

    def loss(self, y_pred, y, current_state_dict, original_state_dict):

        #print(self.proximal_term(current_state_dict, original_state_dict))

        return super().loss(y_pred, y, current_state_dict, original_state_dict) + self.proximal_term(current_state_dict, original_state_dict)
