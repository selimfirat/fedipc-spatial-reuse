import time
from multiprocessing import Process


def objective(device="cuda:0"):

    def _objective(trial):
        from utils import seed_everything
        seed_everything(1)

        from main import main
        import torch

        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1.0, log=True),
            "loss": trial.suggest_categorical("loss", ["l1", "smooth_l1", "mse"]),
            "num_rounds": 500,
            "device": device,
            "mlflow_server": "http://bubota.ee.ic.ac.uk:5000/",
            "input_scaler": "minmax",
            "output_scaler": "minmax"
        }

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
                                storage=f"sqlite:///tmp/optimize_{str(cfg['scenario'])}.db", load_if_exists=True,
                                direction="minimize")
    study.optimize(objective(cfg["device"]), n_trials=cfg["n_trials"])

    return study


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Spatial Reuse with FL - Parameter Study')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=8)

    cfg = vars(parser.parse_args())

    ps = []
    for i in range(cfg["parallel"]):
        p = Process(target=create_worker, args=(cfg,))

        time.sleep(30)
        p.start()

    for p in ps:
        p.join()
