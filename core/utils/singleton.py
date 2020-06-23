from typing import Optional


class SingletonMeta(type):
    ''' Singleton meta class
        Usage: class Sample(metaclass=SingletonMeta):
    '''
    _instance: Optional = None

    def __call__(self):
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance
