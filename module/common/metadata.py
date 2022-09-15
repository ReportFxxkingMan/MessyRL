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


class AbstractValue(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()


class AbstractBuffer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def put(self):
        raise NotImplementedError()
