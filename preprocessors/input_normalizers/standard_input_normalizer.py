

class StandardInputNormalizer:

    def __init__(self, **params):
        self.std = None
        self.mean = None

    def fit(self, X):
        print(X.shape)

        return self

    def transform(self, X):

        return X