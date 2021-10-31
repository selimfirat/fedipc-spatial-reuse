import time
from multiprocessing import Process


def objective(device, mlflow_experiment):

    def _objective(trial):
        from utils import seed_everything
        seed_everything(1)

        from main import main
        import torch

        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1.0, log=True),
            "loss": trial.suggest_categorical("loss", ["l1", "smooth_l1", "mse", "l1-mse"]),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128]),
            "num_epochs": 5,
            "max_num_rounds": 100,
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
            "device": device,
            "mlflow_server": "http://bubota.ee.ic.ac.uk:5000/",
            "mlflow_experiment": mlflow_experiment,
            "input_scaler": trial.suggest_categorical("input_scaler", ["none", "minmax", "standard"]),
            "output_scaler": trial.suggest_categorical("output_scaler", ["none", "minmax"]),
            "federated_trainer": trial.suggest_categorical("federated_trainer", ["fedavg", "fedprox"]),
            "nn_model": "mlp",
        }
        layers = []
        num_layers = trial.suggest_int("num_layers", 1, 4)
        for i in range(num_layers):
            num_hiddens = trial.suggest_categorical(f"num_hiddens_{str(i)}", [16, 32, 64, 128, 256])
            layers.append(num_hiddens)

        params["mlp_hidden_sizes"] = layers

        if params["federated_trainer"] == "fedprox":
            params["mu"] = trial.suggest_float("mu", 1e-4, 1.0, log=True)

        _, eval_val, eval_test = main(params)
        torch.cuda.empty_cache()

        print(params)

        return eval_test["mae"]

    return _objective


def create_worker(cfg):
    import optuna
    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=1)
    study = optuna.create_study(study_name="hyperparameter_study", sampler=sampler,
                                storage=f"sqlite:///tmp/optimize_{cfg['mlflow_experiment']}_{str(cfg['scenario'])}.db", load_if_exists=True,
                                direction="minimize")
    study.optimize(objective(cfg["device"], cfg["mlflow_experiment"]), n_trials=cfg["n_trials"])

    return study


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Spatial Reuse with FL - Parameter Study')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--mlflow_experiment", type=str, default="spatial-reuse-opt1")

    cfg = vars(parser.parse_args())

    ps = []
    for i in range(cfg["parallel"]):
        p = Process(target=create_worker, args=(cfg, ))

        time.sleep(30)
        p.start()

    for p in ps:
        p.join()
