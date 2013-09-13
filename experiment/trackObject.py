#!/usr/bin/env python
import cv2, numpy, os
from operator import itemgetter

class Window:
    def __init__(self, name, fps=30):
        self.name, self.fps = name, fps

    def draw(self, img):
        cv2.imshow(self.name, img)

    def waitKey(self, key):
        return cv2.waitKey(int(1000 / self.fps)) is not ord(key)

class Target:
    def __init__(self, img, keypoints, descriptors):
        self.img, self.keypoints, self.descriptors = img, keypoints, descriptors

class Matcher:
    NN_K        = 2     # The parameter K of K-nearest-neighbor.
    NN_DR       = 0.6   # Nearest-neighbor distance ratio.

    RANSAC_C    = 0.99  # RANSAC confidence parameter.
    RANSAC_D    = 3.0   # RANSAC distance parameter.

    def __init__(self):
        # Use ORB as an alternative to SURF or SIFT.
        self.detector = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def detect(self, img):
        # Convert to gray-scale before running feature detection.
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return self.detector.detectAndCompute(gray_img, None)

    def _filtered_knn_match(self, descriptors_l, descriptors_r):
        try:
            matches = self.matcher.knnMatch(descriptors_l, descriptors_r,
                Matcher.NN_K)
        except:
            return []

        f_nnk = lambda m: len(m) >= 2
        f_nndr = lambda m: m[0].distance >= Matcher.NN_DR * m[1].distance

        return filter(f_nndr, filter(f_nnk, matches))

    def _get_matched_points(self, matches, keypoints_l, keypoints_r):
        points_l, points_r = [], []
        for m in matches:
            points_l.append(keypoints_l[m.queryIdx].pt)
            points_r.append(keypoints_r[m.trainIdx].pt)

        return numpy.array(points_l), numpy.array(points_r)

    def _ransac(self, matches, keypoints_l, keypoints_r, method=cv2.FM_RANSAC):
        points_l, points_r = self._get_matched_points(matches, keypoints_l,
            keypoints_r)

        _, inliers = cv2.findFundamentalMat(points_l, points_r,
            method, Matcher.RANSAC_D, Matcher.RANSAC_C)

        good_matches = []
        for m, is_inliner in zip(matches, inliers):
            if is_inliner:
                good_matches.append(m)

        return good_matches

    def _h_matrix(self, matches, keypoints_l, keypoints_r):
        points_l, points_r = self._get_matched_points(matches, keypoints_l,
            keypoints_r)
        h, inliers = cv2.findHomography(points_r, points_l, cv2.FM_RANSAC)
        inliers = inliers.reshape(1, len(inliers))[0]

        return h, inliers

    def match(self, keypoints, descriptors, target):
        # Use the RobustMatcher method to filter matches.
        # http://stackoverflow.com/a/9894455/881930

        matches_l = self._filtered_knn_match(descriptors, target.descriptors)
        matches_r = self._filtered_knn_match(target.descriptors, descriptors)

        # Keep only symmetric matches.
        matches = []
        for ml in matches_l:
            for mr in matches_r:
                if ml[0].queryIdx == mr[0].trainIdx and \
                        ml[0].trainIdx == mr[0].queryIdx:
                    matches.append(ml[0])

        if len(matches) == 0:
            return None, 0.0, matches

        # Identify good matches using RANSAC.
        good_matches = self._ransac(matches, keypoints, target.keypoints)

        if len(good_matches) == 0:
            return None, 0.0, good_matches

        # Calculate homography matrix.
        h, inliers = self._h_matrix(good_matches, keypoints, target.keypoints)

        return h, inliers, good_matches

class Tracker:
    def __init__(self, image_files, video_file=None):
        if video_file is not None:
            # Convert to absolute path to get around the problem on Mac.
            # http://stackoverflow.com/a/9396912/881930
            video_file = os.path.abspath(video_file)
        else:
            # Use the defaul camera as input.
            video_file = 0

        self.capture = cv2.VideoCapture(video_file)
        self.image_files = map(os.path.abspath, image_files)

        self.matcher = Matcher()
        self.window = Window('Object Tracking Experiment')

    def run(self):
        targets = []
        for image_file in self.image_files:
            img = cv2.imread(image_file)
            keypoints, descriptors = self.matcher.detect(img)

            targets.append(Target(img, keypoints, descriptors))

        while self.window.waitKey('q'):
            ret, img = self.capture.read()

            if not ret:
                break

            keypoints, descriptors = self.matcher.detect(img)

            # Calculate matches for each target image template.
            match_indices = set()
            for target in targets:
                h, confidence, matches = self.matcher.match(keypoints,
                    descriptors, target)

                # Calculate matched indices set.
                for m in matches:
                    match_indices.add(m.queryIdx)

                if h is None:
                    continue

                # Calculate bounding rectangle for matched target.
                width, height, _ = target.img.shape
                rect = numpy.array([[
                    (0, 0), (0, width), (height, width), (height, 0)]],
                    dtype='float32')
                new_rect = cv2.perspectiveTransform(rect, h).astype(int)

                c = (int(numpy.average(map(itemgetter(0), new_rect[0]))),
                    int(numpy.average(map(itemgetter(1), new_rect[0]))))

                # Draw the estimated bounding circle with blue edges.
                cv2.circle(img, c, 10, (255, 0, 0), thickness=5)
                cv2.polylines(img, new_rect, True, (255, 0, 0), thickness=5)

                # Draw the center of matched good points.
                good_points = []
                for m in matches:
                    good_points.append(keypoints[m.queryIdx].pt)
                c = (int(numpy.average(map(itemgetter(0), good_points))),
                    int(numpy.average(map(itemgetter(1), good_points))))
                
                cv2.circle(img, c, 10, (255, 0, 255), thickness=5)

            match_indices = list(match_indices)
            match_indices.sort()

            # Draw matched points in green, and unmatched points in red.
            j = 0
            for i, p in enumerate(keypoints):
                is_matched = False
                if j < len(match_indices):
                    if i >= match_indices[j]:
                        if i == match_indices[j]:
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
    parser.add_argument('image_files', metavar='T', nargs='+',
        help='target object image names')
    parser.add_argument('-f', '--video_file', default=None,
        help='video file name, using the default camera if no file is given')

    args = parser.parse_args()

    # Run motion tracking.
    tracker = Tracker(args.image_files, args.video_file)
    tracker.run()
