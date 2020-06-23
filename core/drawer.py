from tkinter import Canvas
from simple_settings import settings

from core.image import ROIImage


class CanvasDrawer:

    def __init__(self, canvas: Canvas):
        self.canvas = canvas

    def draw_boundary_box(self, roi_image: ROIImage, label: str = None):
        start_x, start_y, end_x, end_y = roi_image.coordinates
        if not label:
            label = str(roi_image)
        self._draw_box(start_x, start_y, end_x, end_y)
        self._draw_label(start_x, start_y, label)

    def _draw_box(self, start_x, start_y, end_x, end_y):
        self.canvas.create_rectangle(
            start_x, start_y, end_x, end_y,
            outline=settings.LINE_COLOR,
            width=settings.LINE_WIDTH
        )

    def _draw_label(self, start_x, start_y, label: str):
        label_id = self.canvas.create_text(
            start_x+13, start_y,
            anchor="nw", text=label,
            font=(settings.FONT_PRINCIPAL, settings.FONT_SIZE),
            fill=settings.FONT_COLOR
        )
        if settings.FONT_BG_COLOR:
            self.__draw_label_background(label_id)

    def __draw_label_background(self, label_id):
        start_x, start_y, end_x, end_y = self.canvas.bbox(label_id)
        rectangle = self.canvas.create_rectangle(
            start_x-15, start_y-5,
            end_x+15, end_y+5,
            fill=settings.FONT_BG_COLOR,
            outline=settings.FONT_BG_COLOR
        )
        self.canvas.tag_lower(rectangle, label_id)
