from config_loader import ConfigLoader
from evaluator import Evaluator
from logger import Logger
from mapper import Mapper
from utils import seed_everything
import pickle

# Set seed
seed_everything(1)

# Get arguments
cfg_loader = ConfigLoader()
cfg_loader.load_by_cli()
cfg = cfg_loader.get_cfg()

logger = Logger(**cfg)

# Download/Load Data
train_loader, val_loader, test_loader = Mapper.get_data_loaders(cfg["scenario"])
test_scenario_loader = Mapper.get_test_loader()

# Preprocess data
cfg["output_size"] = 1 if cfg["scenario"] == 1 else 4
output_scaler = Mapper.get_output_scaler(cfg["output_scaler"])(**cfg)
preprocessor = Mapper.get_preprocessor(cfg["preprocessor"])(output_scaler_ins=output_scaler, **cfg)
train_loader, val_loader, test_loader, test_scenario_loader, cfg["input_size"], output_scaler = preprocessor.fit_transform(train_loader, val_loader, test_loader, test_scenario_loader)

# Train models
evaluator = Evaluator(output_scaler, cfg["metrics"])
trainer = Mapper.get_federated_trainer(cfg["federated_trainer"])(evaluator, logger, **cfg)


experiment = "0a1b1df8ff854e778800e04fecf71d93"
trainer.model = pickle.load(open(f"tmp/results/{experiment}/model.pkl", "rb"))
output_scaler = pickle.load(open(f"tmp/results/{experiment}/output_scaler.pkl", "rb"))

eval_train = evaluator.calculate(trainer, train_loader, fix_negatives=True)
eval_val = evaluator.calculate(trainer, val_loader, fix_negatives=True)
eval_test = evaluator.calculate(trainer, test_loader, fix_negatives=True)

print("Eval Train", eval_train)
print("Eval Val", eval_val)
print("Eval Test", eval_test)


import re
import os
import numpy as np

_, y_pred = trainer.predict(test_scenario_loader)
y_pred = output_scaler.revert(y_pred)
y_pred_int = {int(k[0]): v.numpy() for k, v in y_pred.items()}
y_pred = {k[0]: v.numpy() for k, v in y_pred.items()}

fnames = [fname for fname in os.listdir("data/simulator_input_files_test") if fname.endswith(".csv")]

res = {}
for fname in fnames:
    scenario, threshold = re.findall('\d+', fname)
    res[fname] = y_pred[scenario]

rpath = os.path.join("tmp/tests")
if not os.path.exists(rpath):
    os.makedirs(rpath)

with open(os.path.join(rpath, experiment + ".pkl"), "wb+") as f:
    pickle.dump(res, f)


res_np = [y_pred_int[x] for x in range(1000)]
print(len(res_np))

print(res_np[69], res["input_nodes_test_s069_c-76.csv"])
print(res_np[119], res["input_nodes_test_s119_c-76.csv"])

np.savetxt(os.path.join(rpath, experiment + ".txt"), np.array(res_np), fmt='%s',delimiter=',')

## val
y_true, y_pred = trainer.predict(test_loader)
y_pred = output_scaler.revert(y_pred)
y_test = output_scaler.revert(y_true)
res = {k[0]: {"y_pred": v.numpy(), "y_true":y_test[k]} for k, v in y_pred.items()}

with open(os.path.join(rpath, experiment + "_val.pkl"), "wb+") as f:
    pickle.dump(res, f)
