from abc import ABC, abstractmethod


class AbstractBaseFederatedTrainer(ABC):

    @abstractmethod
    def train(self, nodes_features, nodes_labels, train_contexts):
        pass

    @abstractmethod
    def train_node(self, model, optimizer, X, y):
        pass

    @abstractmethod
    def aggregate(self, chosen_nodes):
        pass

    @abstractmethod
    def predict(self, nodes_features, target_nodes):
        pass
