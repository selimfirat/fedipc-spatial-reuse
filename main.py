from config_loader import ConfigLoader
from evaluator import Evaluator
from mapper import Mapper
from utils import seed_everything


def main():
    # Set seed
    seed_everything(1)

    # Get arguments
    cfg_loader = ConfigLoader()
    cfg = cfg_loader.load_by_cli()

    # Download/Load Data
    train_loader, test_loader = Mapper.get_data_loaders(cfg["scenario"])

    # Preprocess data
    input_scaler = Mapper.get_scaler(cfg["input_scaler"])(**cfg)
    output_scaler = Mapper.get_scaler(cfg["output_scaler"])(**cfg)
    preprocessor = Mapper.get_preprocessor(cfg["preprocessor"])(input_scaler_ins=input_scaler, output_scaler_ins=output_scaler, **cfg)
    train_loader, test_loader, cfg["input_size"], input_scaler, output_scaler = preprocessor.fit_transform(train_loader, test_loader)

    cfg["output_size"] = 1 if cfg["scenario"] == 1 else 4

    # Train models
    trainer = Mapper.get_federated_trainer(cfg["federated_trainer"])(**cfg)
    trainer.train(train_loader)

    # Evaluate
    evaluator = Evaluator(cfg["metrics"])

    y_true, y_pred = trainer.predict(train_loader)
    y_pred = output_scaler.revert(y_pred)
    y_true = output_scaler.revert(y_true)
    eval_train = evaluator.calculate(y_true, y_pred)
    print("Eval Train", eval_train)

    y_true, y_pred = trainer.predict(test_loader)
    y_pred = output_scaler.revert(y_pred)
    y_true = output_scaler.revert(y_true)
    eval_test = evaluator.calculate(y_true, y_pred)
    print("Eval Test", eval_test)


if __name__ == "__main__":
    main()
