from config_loader import ConfigLoader
from data_loader import DataLoader
from evaluator import Evaluator
from federated_trainers.federated_averaging_trainer import FederatedAveragingTrainer

from model_builder import ModelBuilder
from preprocessor import Preprocessor

# Get arguments
cfg_loader = ConfigLoader()
cfg = cfg_loader.load_by_cli()

# Load Data
data_loader = DataLoader()
preprocessor = Preprocessor(cfg["preprocessor"])
evaluator = Evaluator(cfg["metrics"])

nodes_data, y_true_dict, train_contexts, val_contexts, test_contexts = data_loader.get_data()
num_contexts = len(nodes_data.keys())
num_outputs = 1 # Number of throughputs to be predicted. # TODO: check whether number of outputs is greater than 1.

# Preprocess data
nodes_features, nodes_labels, nodes_thresholds = preprocessor.apply(nodes_data, y_true_dict, train_contexts, val_contexts, test_contexts)

# Run model
model_builder = ModelBuilder(nn_model=cfg["nn_model"], input_size=list(nodes_features.values())[0].shape[1], hidden_size=10)
nn_models = model_builder.instantiate_models(num_instances=num_contexts)

nn_models = zip(nodes_data.keys(), nn_models) # Assign a model per context

trainer = FederatedAveragingTrainer(nn_models=nn_models)

nn_models = trainer.train(nodes_features, nodes_labels, train_contexts, val_contexts)

y_pred_dict = y_true_dict

# Evaluate results
res = evaluator.calculate(y_true_dict, y_pred_dict)

print(res)
