from config_loader import ConfigLoader
from evaluator import Evaluator
from logger import Logger
from mapper import Mapper
from utils import seed_everything


def main(override_cfg = None):
    # Set seed
    seed_everything(1)

    # Get arguments
    cfg_loader = ConfigLoader()
    cfg_loader.load_by_cli()
    cfg_loader.override(override_cfg)
    cfg = cfg_loader.get_cfg()

    logger = Logger(**cfg)

    # Download/Load Data
    train_loader, val_loader, test_loader = Mapper.get_data_loaders(cfg["scenario"])

    # Preprocess data
    cfg["output_size"] = 1 if cfg["scenario"] == 1 else 4
    input_scaler = Mapper.get_scaler(cfg["input_scaler"])(**cfg)
    output_scaler = Mapper.get_scaler(cfg["output_scaler"])(**cfg)
    preprocessor = Mapper.get_preprocessor(cfg["preprocessor"])(input_scaler_ins=input_scaler, output_scaler_ins=output_scaler, **cfg)
    train_loader, val_loader, test_loader, cfg["input_size"], input_scaler, output_scaler = preprocessor.fit_transform(train_loader, val_loader, test_loader)

    # Train models
    evaluator = Evaluator(output_scaler, cfg["metrics"])
    trainer = Mapper.get_federated_trainer(cfg["federated_trainer"])(evaluator, logger, **cfg)
    trainer.train(train_loader, val_loader)

    # Evaluate
    eval_train = evaluator.calculate(trainer, train_loader)
    eval_val = evaluator.calculate(trainer, val_loader)
    eval_test = evaluator.calculate(trainer, test_loader)

    print("Eval Train", eval_train)
    print("Eval Val", eval_val)
    print("Eval Test", eval_test)

    logger.log_metrics({ f"train_{k}": v for k,v in eval_train.items() })
    logger.log_metrics({ f"val_{k}": v for k,v in eval_val.items() })
    logger.log_metrics({ f"test_{k}": v for k,v in eval_test.items() })

    logger.close()

    return eval_train, eval_val, eval_test


if __name__ == "__main__":
    main()
