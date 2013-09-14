#!/usr/bin/env python
import cv2

class Window:
    def __init__(self, name, fps=30):
        self.name, self.fps = name, 30
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def show(self, img):
        self.img = img
        cv2.imshow(self.name, self.img)

    def wait(self):
        return cv2.waitKey(int(1000 / self.fps))

    def snapshot(self, filename):
        return cv2.imwrite(filename, self.img)
