from torch.utils.data import DataLoader

from config_loader import ConfigLoader
from dataset import SRDataset, get_data_loaders
from evaluator import Evaluator
from federated_trainers.federated_averaging_trainer import FederatedAveragingTrainer
from maps import Maps
from utils import seed_everything

def main():
    # Set seed
    seed_everything(1)

    # Get arguments
    cfg_loader = ConfigLoader()
    cfg = cfg_loader.load_by_cli()

    # Download/Load Data
    train_loader, test_loader = get_data_loaders(cfg["scenario"])

    # Preprocess data
    preprocessor = Maps.preprocessors[cfg['preprocessor']](**cfg)
    train_loader, test_loader = preprocessor.fit_transform(train_loader, test_loader)

    cfg["output_size"] = 1 if cfg["scenario"] == 1 else 4

    # Train models
    trainer = FederatedAveragingTrainer(**cfg)
    trainer.train(train_loader)

    # Evaluate
    evaluator = Evaluator(cfg["metrics"])

    y_true, y_pred = trainer.predict(train_loader)
    eval_train = evaluator.calculate(y_true, y_pred)

    print("Eval Train", eval_train)

    y_true, y_pred = trainer.predict(test_loader)
    eval_test = evaluator.calculate(y_true, y_pred)
    print("Eval Test", eval_test)


if __name__ == "__main__":
    main()
