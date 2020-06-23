from threading import Thread
from functools import wraps


def threaded(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Thread:
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper
