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

        self.target = cv2.imread(image_file)
        self.capture = cv2.VideoCapture(video_file)
        self.window = Window('Object Tracking Experiment')

        self.detector = None
        self.matcher = None

    def detect_features(self, img):
        if self.detector is None:
            # Use ORB as an alternative to SURF or SIFT.
            self.detector = cv2.ORB()

        # Convert to gray-scale before running feature detection.
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return self.detector.detectAndCompute(gray_img, None)

    def match_features(self, query, template):
        if self.matcher is None:
            # Use Brute-Force matcher for matching descriptors.
            self.matcher = cv2.BFMatcher()

        # Find two nearest-neighbors.
        return self.matcher.knnMatch(query, template, 2)

    def run(self):
        _, target_descriptors = self.detect_features(self.target)

        while self.window.waitKey('q'):
            ret, img = self.capture.read()

            if not ret:
                break

            points, descriptors = self.detect_features(img)
            matches = self.match_features(descriptors, target_descriptors)

            # Nearest-neighbor distance ratio is set to 0.6.
            match_points = []
            for m in matches:
                if m[0].distance <= 0.6 *  m[1].distance:
                    match_points.append(m[0].queryIdx)
            match_points.sort()

            # Draw matched points in green, and unmatched points in red.
            j = 0
            for i, p in enumerate(points):
                is_matched = False
                if j < len(match_points):
                    if i >= match_points[j]:
                        if i == match_points[j]:
                            is_matched = True
                        j += 1
                x, y = p.pt
                cv2.circle(img, (int(x), int(y)), 2,
                    (0, 255, 0) if is_matched else (0, 0, 255), thickness=-1)

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
