#!/usr/bin/env python
import cv2, numpy

class Window:
    THICKNESS   = 3

    def __init__(self, name, fps=30):
        self.name, self.fps = name, fps
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def clear(self):
        self.img = numpy.zeros((480, 640, 3), numpy.uint8)
        self.original_img = self.img

    def draw_dot(self, dot, color):
        cv2.circle(self.img, dot, 1, color, thickness=1)

    def draw_rectangle(self, rect, color):
        x, y, w, h = rect
        cv2.rectangle(self.img, (x,y), (x+w, y+h), color,
            thickness=Window.THICKNESS)

    def draw_polylines(self, poly, color):
        cv2.polylines(self.img, numpy.array([poly]), True, color,
            thickness=Window.THICKNESS)

    def update(self):
        cv2.imshow(self.name, self.img)
        return cv2.waitKey(int(1000 / self.fps))

    def set_image(self, img):
        self.img = numpy.copy(img)
        self.original_img = numpy.copy(img)

    def snapshot(self, filename):
        return cv2.imwrite(filename, self.original_img)
