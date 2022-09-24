from abc import *


class AbstractAgent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()


class AbstractActor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()


class AbstractCritic(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()


class AbstractActionValue(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def create_model(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def get_action(self):
        raise NotImplementedError()

    @abstractmethod
    def get_optimal_action(self):
        raise NotImplementedError()


class AbstractBuffer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def add(self):
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        raise NotImplementedError()

    @abstractmethod
    def size(self):
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        raise NotImplementedError()
