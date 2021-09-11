from sklearn.metrics import mean_squared_error


class Evaluator:

    def __init__(self, metrics=["mse"]):
        self.metrics = metrics

    def calculate(self, y_true_dict, y_pred_dict):

        y_true = []
        y_pred = []

        for sim in y_true_dict.keys():
            for threshold in y_true_dict[sim]:
                y_true.append(y_true_dict[sim][threshold])
                y_pred.append(y_pred_dict[sim][threshold])

        results = {}

        for metric_name in self.metrics:
            metric_func = getattr(self, metric_name)
            results[metric_name] = metric_func(y_true, y_pred)

        return results

    def mse(self, y_true, y_pred):

        return mean_squared_error(y_true, y_pred)
