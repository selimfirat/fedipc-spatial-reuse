class Mapper:

    @staticmethod
    def get_federated_trainer(federated_trainer):
        from federated_trainers.federated_averaging_trainer import FederatedAveragingTrainer

        return {
            "fed_avg": FederatedAveragingTrainer
        }[federated_trainer]

    @staticmethod
    def get_loss(loss):
        from torch import nn

        return {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "smooth_l1": nn.SmoothL1Loss
        }[loss]

    @staticmethod
    def get_nn_model(nn_model):
        from nn_models.mlp import MLP
        from nn_models.separate_recurrents import SeparateRecurrentsModel

        return {
            "mlp": MLP,
            "separate_recurrents": SeparateRecurrentsModel,
        }[nn_model]

    @staticmethod
    def get_preprocessor(preprocessor):
        from preprocessors.basic_features_preprocessor import BasicFeaturesPreprocessor
        from preprocessors.sequential_features_preprocessor import SequentialFeaturesPreprocessor

        return {
            "basic_features": BasicFeaturesPreprocessor,
            "sequential_features": SequentialFeaturesPreprocessor
        }[preprocessor]

    @staticmethod
    def get_data_loaders(scenario):
        from dataset import SRDataset, DataDownloader
        from torch.utils.data import DataLoader

        data_downloader = DataDownloader(scenario)

        train_data = SRDataset(data_downloader, is_train=True)
        test_data = SRDataset(data_downloader, is_train=False)

        train_loader = DataLoader(train_data)
        test_loader = DataLoader(test_data)

        return train_loader, test_loader