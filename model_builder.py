from nn_models import modelname_2_modelcls


class ModelBuilder:

    def __init__(self, nn_model, **params):
        self.nn_model = nn_model
        self.params = params
        self.nn_model_cls = modelname_2_modelcls[self.nn_model]

    def instantiate_model(self):

        model = self.nn_model_cls(**self.params)

        return model
