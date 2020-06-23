import tkinter as tk
from PIL.Image import Image
from PIL.ImageTk import PhotoImage

from core.window import AbstractWindow
from core.drawer import CanvasDrawer


class FaceMaskRecognitionWindow(AbstractWindow, tk.Frame):

    canvas: tk.Canvas
    drawer: CanvasDrawer
    __last_canvas_image: PhotoImage

    def __init__(self, window_title="Frame"):
        self.window_title = window_title
        super().__init__()

    def open(self):
        self.winfo_toplevel().title(self.window_title)
        self.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.drawer = CanvasDrawer(self.canvas)

    def close(self):
        self.destroy()

    def render(self, frame: Image, detected_objects):
        self._clean()
        self._draw_frame(frame)
        self._draw_boundary_boxes(detected_objects)

    def resize(self, hight: int, width: int):
        self.winfo_toplevel().geometry(f"{hight}x{width}")

    def _clean(self):
        self.canvas.delete("all")

    def _draw_frame(self, frame: Image):
        # store the last canvas image,
        # otherwise it will be removed by the garbage collector and will not be displayed
        self.__last_canvas_image = PhotoImage(image=frame)
        self.canvas.create_image(0, 0, image=self.__last_canvas_image, anchor=tk.NW)

    def _draw_boundary_boxes(self, detected_objects):
        for detected_object in detected_objects:
            self.drawer.draw_boundary_box(detected_object.roi_image, str(detected_object))
