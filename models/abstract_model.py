import abc


class AbstractModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, X, y):

        raise NotImplementedError
