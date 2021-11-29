# fedipc-spatial-reuse
FedIPC Spatial Reuse Project for ITU AI/ML Challenge 2021.

## Team Members
* Selim Firat Yilmaz
* Ahmet Kerem Ozfatura
* Mehmet Emre Ozfatura
* Ozlem Yildiz
* Mentor: Prof. Deniz Gunduz

## Setup
* `pip install requirements.txt`
* Running using the command below automatically donwloads the data

## Running
* `python main.py`
* or you may use arguments via `python main.py --scenario 1 --preprocessor all_features` 

## Command Line  Interface Arguments
* Run `python main.py --help` for details.
* `--scenario` (default: `1`)
    * `1` or `2`
* `--preprocessor` (default: `mean_features`)
    * `mean_features`
* `--nn_model` (default: `mlp`)
    * `mlp`
* `--fed_model` (default: `fed_avg`)
    * `fed_avg`
* `--metrics` (can include multiple metrics, default: `mse r2`)
    * `mse`
    * `r2`

## Extending

### Adding New Neural Network Model
1. Create a model extending `torch.nn.Module` in `./nn_models` directory.
2. Update `./mapper.py`.
3. Use the new model with `--nn_model $model_name` cli argument.

### Adding New Federated Learning Architecture
1. Create a trainer class extending `federated_trainers.abstract_base_federated_trainer.AbstractBaseFederatedTrainer` in `./federated_trainers` directory.
2. Update `./mapper.py`.
3. Use the new FL architecture with `--fed_model $trainer_name` cli argument.

### Adding New Preprocessor (Feature Extractor)
1. Create a preprocessor class extending `federated_trainers.abstract_base_preprocessor.AbstractBasePreprocessor` in `./preprocessors` directory.
2. Update `./mapper.py`.
3. Use the new preprocessor with `--preprocessor $new_preprocessor_name`.

### Adding New CLI Argument
1. Add the argparse argument via `parser.add_argument` in `./config_loader.py` file. Tutorial: [this link](https://www.pythonforbeginners.com/argparse/argparse-tutorial).

## Cleaning
Data loader and preprocessors cache once applied. Thus, these cached are needed to be cleaned once these methods are modified.
* `bash scripts/clear_tmp.sh`, where working directory is `fedipc-spatial-reuse`.