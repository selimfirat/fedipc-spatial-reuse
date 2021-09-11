from config_loader import ConfigLoader
from data_loader import DataLoader
from evaluator import Evaluator
from preprocessor import Preprocessor


# Get arguments
cfg_loader = ConfigLoader()
cfg = cfg_loader.load_by_cli()

# Load Data
data_loader = DataLoader()

nodes_data, y_true_dict = data_loader.get_data()
# TODO: split data to training, validation, test splits

# Preprocess data
preprocessor = Preprocessor(cfg["preprocessor"])
nodes_features, nodes_labels = preprocessor.apply(nodes_data)

# Run model
y_pred_dict = y_true_dict


# Evaluate results
evaluator = Evaluator(cfg["metrics"])

res = evaluator.calculate(y_true_dict, y_pred_dict)

print(res)
