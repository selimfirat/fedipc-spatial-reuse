

class DummyInputNormalizer:

    def __init__(self, **params):
        pass

    def fit(self, X):
        print(X.shape)

        return self

    def transform(self, X):

        return X
