from nn_models.mlp import MLP
from nn_models.separate_recurrents import SeparateRecurrentsModel

nn_model_map = {
    "mlp": MLP,
    "separate_recurrents": SeparateRecurrentsModel,
}

def get_nn_model(cfg):

        return nn_model_map[cfg['nn_model']](**cfg)
