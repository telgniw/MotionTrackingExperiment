#!/usr/bin/env python
#-*- coding: utf8 -*-
from lib.cli import CmdRunner

from lib.camera import Camera
from lib.window import Window

from lib.tracking import *

import sys

class Task:
    def __init__(self, method, args):
        self.method, self.args = method, args

class Main:
    def __init__(self):
        self.tasks = []

        self.camera = Camera()
        self.window = Window('Camera')

        self.detector = Detector()

    def loop(self):
        runner = CmdRunner(self)
        runner.start()
        
        while runner.isAlive():
            while len(self.tasks) > 0:
                task = self.tasks.pop(0)
                task.method(*task.args)

            try:
                img = self.camera.get_frame()

                if img is None:
                    self.window.clear()
                    continue

                self.window.set_image(img)

                rects = self.detector.detect(img)
                for rect in rects:
                    self.window.draw_polylines(rect, (255, 0, 0))
            except Exception as e:
                print >> sys.stderr, e
            finally:
                self.window.update()
                runner.join(0)

    def do_add_target(self, filename):
        self.tasks.append(Task(self.detector.add_image, (filename, )))

    def do_snapshot(self, filename):
        self.tasks.append(Task(self.window.snapshot, (filename, )))

    def do_switch_camera(self, cid):
        self.tasks.append(Task(self.camera.switch, (cid, )))

if __name__ == '__main__':
    Main().loop()
