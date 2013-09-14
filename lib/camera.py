#!/usr/bin/env python
import cv2

class Camera:
    def get_frame(self):
        if self.capture is None:
            return None

        ret, img = self.capture.read()
        return img if ret else None

    def set_id(self, cid):
        if cid is None:
            self.capture = None
        else:
            self.capture = cv2.VideoCapture(cid)
