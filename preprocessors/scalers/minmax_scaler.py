import torch


class MinMaxScaler:

    def __init__(self, **params):
        pass

    def fit(self, X):

        rt = torch.stack(X, dim=0)
        self.min = rt.amin(dim=[0, 1])
        self.max = rt.amax(dim=[0, 1])

        self.diff = self.max - self.min
        return self

    def transform(self, X):

        X = [(x - self.min) / self.diff for x in X]

        return X

    def revert(self, labels):

        new_labels = {}
        for context_idx, context_labels in labels.items():
            new_labels[context_idx] = (context_labels * self.diff) + self.min

        return new_labels
