

class DummyScaler:

    def __init__(self, **params):
        pass

    def fit(self, X):

        return self

    def transform(self, X):

        return X

    def revert(self, labels):

        return labels
