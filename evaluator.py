from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch


# TODO: Implement a metric for fairness since the aim is "increasing performance while reducing the number of undesired situations (e.g. poor fairness)."
class Evaluator:

    def __init__(self, output_scaler, metrics=None):
        if metrics is None:
            metrics = ["mse", "r2", "mae"]
        self.metrics = metrics
        self.output_scaler = output_scaler

    def calculate(self, trainer, data_loader, fix_negatives=False):
        y_true, y_pred = trainer.predict(data_loader)
        y_pred = self.output_scaler.revert(y_pred)
        y_true = self.output_scaler.revert(y_true)

        def _calculate(y_true_dict, y_pred_dict):
            y_true = torch.cat([y_true_dict[context_idx] for context_idx in y_pred_dict.keys()], dim=0).numpy().flatten()
            y_pred = torch.cat([y_pred_dict[context_idx] for context_idx in y_pred_dict.keys()], dim=0).numpy().flatten()

            if fix_negatives:
                y_pred[y_pred < 0.0] = 0.0
            
            results = {}

            for metric_name in self.metrics:
                metric_func = getattr(self, metric_name)
                results[metric_name] = metric_func(y_true, y_pred)

            return results

        eval_res = _calculate(y_true, y_pred)

        return eval_res

    def mse(self, y_true, y_pred):

        return mean_squared_error(y_true, y_pred)

    def mae(self, y_true, y_pred):

        return mean_absolute_error(y_true, y_pred)

    def r2(self, y_true, y_pred):

        return r2_score(y_true, y_pred)
