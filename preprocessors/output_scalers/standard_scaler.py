import torch


class StandardScaler:

    def __init__(self, **params):
        self.std = None
        self.mean = None

    def fit(self, X):

        if isinstance(X[0][0], dict):
            return self.fit_dict(X)
        else:
            return self.fit_list(X)

    def transform(self, X):

        if isinstance(X[0][0], dict):
            return self.transform_dict(X)
        else:
            return self.transform_list(X)

    def fit_list(self, X):
        rt = torch.stack(X, dim=0)
        self.mean = rt.mean()
        self.std = rt.std()

        self.std[self.std == 0.0] = 1.0

        return self

    def fit_dict(self, X):

        R = { k: [] for k in X[0][0].keys() }
        for xx in X:
            for xxx in xx:
                for k in xxx.keys():
                    R[k].append(xxx[k])

        self.mean = {}
        self.std = {}
        for key in R.keys():
            rt = torch.stack(R[key], dim=0)
            self.mean[key] = rt.mean()
            self.std[key] = rt.std()
            self.std[key][self.std[key] == 0.0] = 1.0

        return self

    def transform_list(self, X):

        R = [(x - self.mean) / self.std for x in X]

        return R

    def transform_dict(self, X):

        res = []

        for x in X:
            cur_res = []
            for xx in x:
                R = {k: [] for k in X[0][0].keys()}
                for key in xx.keys():
                    R[key] = (xx[key] - self.mean[key]) / self.std[key]

                cur_res.append(R)
            res.append(cur_res)

        return res

    def revert(self, labels):

        new_labels = {}
        for context_idx, context_labels in labels.items():
            new_labels[context_idx] = (context_labels * self.std) + self.mean

        return new_labels
