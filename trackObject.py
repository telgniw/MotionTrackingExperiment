#!/usr/bin/env python
import cv2, numpy, os
from random import randint, shuffle

class Window:
    def __init__(self, name, fps=30):
        self.name, self.fps = name, fps

    def draw(self, img):
        cv2.imshow(self.name, img)

    def waitKey(self, key):
        return cv2.waitKey(int(1000 / self.fps)) is not ord(key)

class Tracker:
    def __init__(self, image_file, video_file=None):
        if video_file is not None:
            # Convert to absolute path to get around the problem on Mac.
            # http://stackoverflow.com/a/9396912/881930
            video_file = os.path.abspath(video_file)
        else:
            # Use the defaul camera as input.
            video_file = 0

        self.capture = cv2.VideoCapture(video_file)
        self.window = Window('Object Tracking Experiment')

    def run(self):
        while self.window.waitKey('q'):
            ret, img = self.capture.read()

            if not ret:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Use ORB as an alternative to SURF or SIFT.
            detector = cv2.ORB()
            points = detector.detect(gray_img)

            for i, p in enumerate(points):
                x, y = p.pt
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), thickness=-1)

            self.window.draw(img)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Object tracking.')
    parser.add_argument('image_file', metavar='T',
        help='target object image name')
    parser.add_argument('-f', '--video_file', default=None,
        help='video file name, using the default camera if no file is given')

    args = parser.parse_args()

    # Run motion tracking.
    tracker = Tracker(args.image_file, args.video_file)
    tracker.run()
