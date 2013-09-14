#!/usr/bin/env python
import cv2

class Camera:
    def __init__(self):
        self.cid, self._update = None, False

    def _update(self):
        if not self._update:
            return

        if self.cid is None:
            self.capture = None
        else:
            self.capture = cv2.VideoCapture(self.cid)

        self._update = False

    def get_frame(self):
        self._update()

        if self.capture is None:
            return None

        ret, img = self.capture.read()
        return img if ret else None

    def set_id(self, cid):
        self.cid, self._update = cid, True
