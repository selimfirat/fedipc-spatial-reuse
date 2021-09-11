

class Preprocessor:

    def __init__(self, preprocess_type="all_features"):
        self.preprocess_type = preprocess_type

    def apply(self, data):
        func = getattr(self, self.preprocess_type)

        return func(data)

    def all_features(self, data):
        # TODO: fill
        pass
