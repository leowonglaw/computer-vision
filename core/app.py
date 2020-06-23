import logging
import tkinter as tk

from .utils.singleton import SingletonMeta

LOG = logging.getLogger(__name__)


class TkinterRoot(tk.Tk, metaclass=SingletonMeta):
    pass


class GUIApp:

    def __init__(self):
        LOG.info('starting')
        self.root = TkinterRoot()
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def serve(self):
        LOG.info('started')
        self.root.mainloop()
        LOG.info('stopped')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            LOG.critical("starting failed",
                         exc_info=(exc_type, exc_val, exc_tb))
            return True
        self.serve()
