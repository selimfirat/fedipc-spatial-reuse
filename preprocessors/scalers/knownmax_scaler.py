from preprocessors.scalers.minmax_scaler import MinMaxScaler


class KnownMaxScaler(MinMaxScaler):

    def fit(self, X):

        self.max = 135.0
        self.min = 0.0

        self.diff = self.max - self.min

        return self
