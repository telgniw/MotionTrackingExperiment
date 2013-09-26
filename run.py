#!/usr/bin/env python
from lib.cli import CmdRunner

from lib.camera import Camera
from lib.window import Window

from lib.filtering import *
from lib.tracking import *

import sys

class Task:
    def __init__(self, method=None, args=tuple()):
        self.method, self.args = method, args

    def execute(self):
        if self.method is None:
            return
        self.method(*self.args)

class Main:
    def __init__(self):
        self.tasks = []

        self.camera = Camera()
        self.window = Window('Camera')

        self.should_detect = True
        self.trackers = []

        self.color_filter = ColorFilter()

    def add_target(self, filename):
        self.trackers.append(Tracker(filename))

    def clear(self):
        for tracker in self.trackers:
            tracker.clear()

    def loop(self):
        runner = CmdRunner(self)
        runner.start()
        
        while runner.isAlive():
            while len(self.tasks) > 0:
                task = self.tasks.pop(0)
                task.execute()

            try:
                img = self.camera.get_frame()

                if img is None:
                    self.window.clear()
                    continue

                img = cv2.resize(img, (640, 480))

                # TODO: show color mask
                color_mask = self.color_filter.filter(img)

                next_should_detect = self.should_detect
                for tracker in self.trackers:
                    if self.should_detect:
                        rect = tracker.detect(img)

                        if rect is not None:
                            ret = tracker.update_window(img, rect)
                            self.window.draw_rectangle(rect, (255, 0, 255))
                        next_should_detect = False

                    rect = tracker.track(img)

                    if rect is None:
                        next_should_detect = True
                        continue

                    self.window.draw_rectangle(rect, (255, 0, 0))

                self.should_detect = next_should_detect
            except Exception as e:
                print >> sys.stderr, e
            finally:
                self.window.update()
                runner.join(0)

    def _add_task(self, task):
        self.should_detect = True
        self.tasks.append(task)

    def do_add_target(self, filename):
        self._add_task(Task(self.add_target, (filename, )))

    def do_clear(self):
        self._add_task(Task(self.clear))

    def do_snapshot(self, filename):
        self._add_task(Task(self.window.snapshot, (filename, )))

    def do_switch_camera(self, cid):
        self._add_task(Task(self.camera.switch, (cid, )))

if __name__ == '__main__':
    Main().loop()
