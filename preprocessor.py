import numpy as np


class Preprocessor:

    def __init__(self, preprocess_type="basic_features"):
        self.preprocess_type = preprocess_type

    def apply(self, nodes):
        func = getattr(self, self.preprocess_type)

        features = {sim_no: {} for sim_no in nodes.keys()}
        for sim_no, node in nodes.items():
            features[sim_no] = func(node)

        return features

    def basic_features(self, node_data):

        r = ["interference", "rssi", "sinr"]

        res = np.empty((len(r),))
        for i in range(len(r)):
            res[i] = np.mean(node_data[r[i]])

        return res
