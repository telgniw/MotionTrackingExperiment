#!/usr/bin/env python
#-*- coding: utf8 -*-
from lib.camera import Camera
from lib.window import Window

from lib.cli import CmdRunner

class Task:
    def __init__(self, method, args):
        self.method, self.args = method, args

class Main:
    def __init__(self):
        self.tasks = []

        self.camera = Camera()
        self.window = Window('Camera')

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
                    continue

                self.window.show(img)
            except:
                pass
            finally:
                self.window.wait()
                runner.join(0)

    def set_camera_id(self, cid):
        self.tasks.append(Task(self.camera.set_id, (cid, )))

    def snapshot(self, filename):
        self.tasks.append(Task(self.window.snapshot, (filename, )))

if __name__ == '__main__':
    Main().loop()
