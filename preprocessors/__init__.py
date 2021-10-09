from torch.utils.data import Dataset

from preprocessors.basic_features_preprocessor import BasicFeaturesPreprocessor
from preprocessors.sequential_features_preprocessor import SequentialFeaturesPreprocessor

preprocessor_map = {
    "basic_features": BasicFeaturesPreprocessor,
    "sequential_features": SequentialFeaturesPreprocessor
}


def get_preprocessor(cfg):

    return preprocessor_map[cfg['preprocessor']](**cfg)
