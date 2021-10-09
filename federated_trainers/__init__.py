from federated_trainers.federated_averaging_trainer import FederatedAveragingTrainer
from torch import nn

modelname_2_modelcls = {
    "fed_avg": FederatedAveragingTrainer
}

loss_map = {
    "mse": lambda x: nn.MSELoss(*x),
    "l1": lambda x: nn.L1Loss(*x),
    "smooth_l1": lambda x: nn.SmoothL1Loss(*x)
}

def get_loss(loss):

    return loss_map[loss]
