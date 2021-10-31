import mlflow
import os
import dill as pickle

class Logger:

    def __init__(self, mlflow_server, mlflow_experiment, **params):

        self.log = mlflow_server is not None

        if self.log:
            mlflow.set_tracking_uri(mlflow_server)
            mlflow.set_experiment(mlflow_experiment)
            cur_run = mlflow.start_run()
            self.run_id = cur_run.info.run_id

            for param, val in params.items():
                self.log_param(param, val)

    def log_metric(self, name, val, step=0):
        print(step, name, val)
        if self.log:
            mlflow.log_metric(name, val, step)

    def log_metrics(self, metrics_dict):
        if self.log:
            mlflow.log_metrics(metrics_dict)

    def log_param(self, name, val):
        if self.log:
            mlflow.log_param(name, val)

    def close(self):
        if self.log:
            mlflow.end_run()

    def log_artifacts(self, artifacts):
        if self.log:
            path = f"tmp/results/{self.run_id}"
            if not os.path.exists(path):
                os.makedirs(path)
            
            for name, item in artifacts.items():
                rpath = os.path.join(path, f"{name}.pkl")
                with open(rpath, "wb+") as f:
                    pickle.dump(item, f)

            mlflow.log_artifacts(path)