from config_loader import ConfigLoader
from data_loader import DataLoader
from evaluator import Evaluator
from federated_trainers.federated_averaging_trainer import FederatedAveragingTrainer
from utils import seed_everything
from model_builder import ModelBuilder
from preprocessor import Preprocessor

def main():
    # Set seed
    seed_everything(1)

    # Get arguments
    cfg_loader = ConfigLoader()
    cfg = cfg_loader.load_by_cli()

    # Load Data
    data_loader = DataLoader(cfg["scenario"])
    nodes_data, y_true_dict, train_contexts, test_contexts = data_loader.get_data()

    # Preprocess data
    preprocessor = Preprocessor(cfg["preprocessor"], cfg["scenario"])
    nodes_features, nodes_labels, nodes_thresholds = preprocessor.apply(nodes_data, y_true_dict, train_contexts, test_contexts)

    # Build models
    input_size = list(nodes_features.values())[0].shape[1]
    model_builder = ModelBuilder(nn_model=cfg["nn_model"], input_size=input_size, hidden_size=10)
    model = model_builder.instantiate_model()

    # Train models
    trainer = FederatedAveragingTrainer(model, num_rounds=1, participation=1.0, num_epochs=5, lr=1e-3, batch_size=16)

    trainer.train(nodes_features, nodes_labels, train_contexts)

    # Train metrics
    y_pred = trainer.predict(nodes_features, train_contexts)
    y_pred_dict = Evaluator.build_pred_dict(y_pred, train_contexts, nodes_thresholds)

    # Evaluate results
    evaluator = Evaluator(cfg["metrics"])

    res = evaluator.calculate(y_true_dict, y_pred_dict)

    print("Training", res)

    # Validation metrics
    y_pred = trainer.predict(nodes_features, test_contexts)
    y_pred_dict = Evaluator.build_pred_dict(y_pred, test_contexts, nodes_thresholds)

    # Evaluate results
    evaluator = Evaluator(cfg["metrics"])

    res = evaluator.calculate(y_true_dict, y_pred_dict)

    print("Validation", res)


if __name__ == "__main__":
    main()
