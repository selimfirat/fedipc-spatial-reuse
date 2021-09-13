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
    data_loader = DataLoader()
    nodes_data, y_true_dict, train_contexts, val_contexts, test_contexts = data_loader.get_data()
    num_contexts = len(nodes_data.keys())
    num_outputs = 1 # Number of throughputs to be predicted. # TODO: check whether number of outputs/throughputs is greater than 1.

    # Preprocess data
    preprocessor = Preprocessor(cfg["preprocessor"])
    nodes_features, nodes_labels, nodes_thresholds = preprocessor.apply(nodes_data, y_true_dict, train_contexts, val_contexts, test_contexts)

    # Build models
    model_builder = ModelBuilder(nn_model=cfg["nn_model"], input_size=list(nodes_features.values())[0].shape[1], hidden_size=10)
    main_model, nn_models = model_builder.instantiate_models(num_instances=len(train_contexts))

    nn_models = dict(zip(train_contexts, nn_models)) # Assign a model per context

    # Train models
    trainer = FederatedAveragingTrainer(main_model=main_model, nn_models=nn_models, num_rounds=3, participation=1.0, num_epochs=5, lr=1e-3, batch_size=8)

    nn_models = trainer.train(nodes_features, nodes_labels, train_contexts, val_contexts)

    y_pred_dict = y_true_dict

    # Evaluate results
    evaluator = Evaluator(cfg["metrics"])

    res = evaluator.calculate(y_true_dict, y_pred_dict)

    print(res)

if __name__ == "__main__":
    main()