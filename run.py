#!/usr/bin/env python
#-*- coding: utf8 -*-
from lib.camera import Camera
from lib.window import Window

from lib.cli import CmdRunner

class Main:
    def __init__(self):
        self.camera = Camera()
        self.window = Window('Camera')

    def loop(self):
        runner = CmdRunner(self)
        runner.start()
        
        while runner.isAlive():
            try:
                img = self.camera.get_frame()

                if img is not None:
                    self.window.show(img)
            except:
                pass

            runner.join(0)

    def set_camera_id(self, cid):
        self.camera.set_id(cid)

if __name__ == '__main__':
    Main().loop()
