from abc import ABC, abstractmethod


class AbstractWindow(ABC):

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass
