from argparse import ArgumentParser


class ConfigLoader:

    def load_by_cli(self):
        parser = ArgumentParser(description="FedIPC Spatial Reuse Project for ITU AI/ML Challenge 2021.")

        parser.add_argument("--model", default="fed_avg", type=str, help="Model name")
        parser.add_argument("--preprocessor", default="basic_features", type=str,
                            help="Preprocessor function applied to the raw data")
        parser.add_argument("--metrics", type=str, nargs='+', default=["mse"], help="List of metrics to be calculated")

        cfg = vars(parser.parse_args())

        return cfg
