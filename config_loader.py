from argparse import ArgumentParser


class ConfigLoader:

    def load_by_cli(self):
        parser = ArgumentParser(description="FedIPC Spatial Reuse Project for ITU AI/ML Challenge 2021.")

        parser.add_argument("--mlflow_server", type=str, default="http://bubota.ee.ic.ac.uk:5000/")
        parser.add_argument("--mlflow_experiment", type=str, default="spatial-reuse")

        parser.add_argument("--scenario", default=3, type=int, help="Scenario number (1 or 2 at https://zenodo.org/record/5506248#.YVMaMUZBxpR) ")
        parser.add_argument("--federated_trainer", default="fedavg", type=str, help="Federated Architecture")
        parser.add_argument("--nn_model", default="mlp", type=str, help="NN Model")
        parser.add_argument("--preprocessor", default="padded_features", type=str, help="Preprocessor applied to the raw data")
        parser.add_argument("--input_scaler", default="minmax", type=str, help="Normalizer applied to the preprocessed data")
        parser.add_argument("--output_scaler", default="minmax", type=str, help="Normalizer applied to the labels")
        parser.add_argument("--metrics", type=str, nargs='+', default=["mse", "r2", "mae"], help="List of metrics to be calculated")
        parser.add_argument("--device", type=str, default="cpu")

        # Federated Trainer Params
        parser.add_argument("--participation", type=float, default=1.0)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--max_num_rounds", type=int, default=1000)
        parser.add_argument("--early_stopping_patience", type=int, default=5)
        parser.add_argument("--early_stopping_check_rounds", type=int, default=20)
        parser.add_argument("--loss", type=str, default="mse",  help="mse, l1, smooth_l1")
        parser.add_argument("--optimizer", type=str, default="sgd",  help="sgd, adam, adamw")

        # Local SGD Parameters
        parser.add_argument("--lr", type=float, default=0.5)
        parser.add_argument("--momentum", type=float, default=0.0)
        parser.add_argument("--nesterov", type=float, default=False)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--dampening", type=float, default=0.0)

        # FedProx params
        parser.add_argument("--mu", type=float, default=1e-2, help="Fedprox proximal term multiplier mu.")

        ## NN Model Parameters
        # MLP Parameters
        parser.add_argument("--mlp_hidden_sizes", type=list, default=[256])
        parser.add_argument("--mlp_activation", type=str, default="relu")
        ## Transformer
        parser.add_argument("--nhead", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--dropout",type=float,default=0.4)

        parser.add_argument("--parallel", type=int, default=1)

        self.cfg = vars(parser.parse_args())

        return self.cfg

    def override(self, override_cfg):
        if override_cfg is None:
            return self.cfg

        for k, v in override_cfg.items():
            self.cfg[k] = v

        return self.cfg

    def get_cfg(self):

        return self.cfg