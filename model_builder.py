from nn_models import modelname_2_modelcls


class ModelBuilder:

    def __init__(self, nn_model, **params):
        self.nn_model = nn_model
        self.params = params
        self.nn_model_cls = modelname_2_modelcls[self.nn_model]

    def instantiate_models(self, num_instances):
        models = []
        for i in range(num_instances):
            model = self.nn_model_cls(**self.params)
            models.append(model)

        main_model = self.nn_model_cls(**self.params)

        return main_model, models
