def objective(trial):
    from utils import seed_everything
    seed_everything(1)

    from main import main
    import torch

    params = {}

    params["lr"] = trial.suggest_loguniform("lr", 1e-5, 1.0)
    params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 21])

    _, eval_test = main(params)
    torch.cuda.empty_cache()

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
    study = optuna.create_study(study_name="hyperparameter_study", sampler=sampler, storage=f"sqlite:///figures/optimize_{str(cfg['scenario'])}.db", load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=100)
