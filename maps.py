from torch import nn
from nn_models.mlp import MLP
from nn_models.separate_recurrents import SeparateRecurrentsModel
from preprocessors.basic_features_preprocessor import BasicFeaturesPreprocessor
from preprocessors.sequential_features_preprocessor import SequentialFeaturesPreprocessor


class MapInstance:

    losses = {
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss
    }

    nn_models = {
        "mlp": MLP,
        "separate_recurrents": SeparateRecurrentsModel,
    }

    preprocessors = {
        "basic_features": BasicFeaturesPreprocessor,
        "sequential_features": SequentialFeaturesPreprocessor
    }


Maps = MapInstance()
