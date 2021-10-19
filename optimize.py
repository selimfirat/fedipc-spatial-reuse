def objective(trial):
    from utils import seed_everything
    seed_everything(1)

    from main import main
    import torch

    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1.0, log=True),
        "loss": trial.suggest_categorical("loss", ["l1", "smooth_l1", "mse"]),
        "num_rounds": 500,
        "device": "cuda:0"
    }

    _, eval_test = main(params)
    torch.cuda.empty_cache()

    print(params)

    return eval_test["mae"]


if __name__ == '__main__':
    import argparse
    import optuna
    from optuna.samplers import TPESampler

    parser = argparse.ArgumentParser(description='Spatial Reuse with FL - Parameter Study')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scenario", type=int, default=1)

    cfg = vars(parser.parse_args())

    sampler = TPESampler(seed=1)
    study = optuna.create_study(study_name="hyperparameter_study", sampler=sampler, storage=f"sqlite:///tmp/optimize_{str(cfg['scenario'])}.db", load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=10)
