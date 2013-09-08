#!/usr/bin/env python
import cv, cv2, numpy, os, sys

class Window:
    def __init__(self, window_name, fps):
        self.name = window_name
        self.fps = fps

        cv2.namedWindow(self.name, cv.CV_WINDOW_AUTOSIZE)

    def draw(self, img):
        cv2.imshow(self.name, img)

        # Wait for 1000/fps milliseconds to prevent window from blocking.
        cv2.waitKey(int(1000 / self.fps));

class Tracker:
    def __init__(self, video_file=None):
        if video_file is not None:
            # Convert to absolute path to get around the problem on Mac.
            # http://stackoverflow.com/a/9396912/881930
            video_file = os.path.join(os.getcwd(), video_file)
        else:
            # Set video input to the first camera if no file is specified.
            video_file = 0

        self.capture = cv2.VideoCapture(video_file)

        # Get video information.
        self.length = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
        self.fps    = self.capture.get(cv.CV_CAP_PROP_FPS)
        self.width  = self.capture.get(cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

        # Create display window.
        self.window = Window('Motion Tracking Experiment', self.fps)

    def run(self):
        for i in range(self.length):
            ret, img = self.capture.read()

            if not ret:
                break

            self.window.draw(img)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--video_file', default=None,
        help='video file name')

    args = parser.parse_args()

    # Run motion tracking.
    tracker = Tracker(args.video_file)
    tracker.run()
