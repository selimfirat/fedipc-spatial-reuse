from argparse import ArgumentParser


class ConfigLoader:

    def load_by_cli(self):
        parser = ArgumentParser(description="FedIPC Spatial Reuse Project for ITU AI/ML Challenge 2021.")

        parser.add_argument("--scenario", default=1, type=int, help="Scenario number (1 or 2 at https://zenodo.org/record/5506248#.YVMaMUZBxpR) ")
        parser.add_argument("--federated_trainer", default="fed_avg", type=str, help="Federated Architecture")
        parser.add_argument("--nn_model", default="mlp", type=str, help="NN Model")
        parser.add_argument("--preprocessor", default="mean_features", type=str, help="Preprocessor applied to the raw data")
        parser.add_argument("--input_scaler", default="standard", type=str, help="Normalizer applied to the preprocessed data")
        parser.add_argument("--output_scaler", default="standard", type=str, help="Normalizer applied to the labels")
        parser.add_argument("--metrics", type=str, nargs='+', default=["mse", "r2"], help="List of metrics to be calculated")
        parser.add_argument("--device", type=str, default="cpu")

        # Federated Trainer Params
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--participation", type=float, default=1.0)
        parser.add_argument("--batch_size", type=int, default=21)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--num_rounds", type=int, default=100)
        parser.add_argument("--loss", type=str, default="mse",  help="mse, l1, smooth_l1")

        ## NN Model Parameters
        # MLP Parameters
        parser.add_argument("--mlp_hidden_sizes", type=list, default=[50, 25])
        parser.add_argument("--mlp_activation", type=str, default="relu")

        cfg = vars(parser.parse_args())

        return cfg
