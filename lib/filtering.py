#!/usr/bin/env python
import cv2

class ColorFilter:
    def __init__(self):
        self.color = ((110, 150, 200), (130, 200, 255))

    def filter(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img, self.color[0], self.color[1])
        return cv2.GaussianBlur(mask, (5, 5), 0)

    def set_color_range(self, min_color, max_color):
        self.color = (min_color, max_color)
