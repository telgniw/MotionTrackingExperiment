#!/usr/bin/env python
import cv2, numpy
import util

class Target:
    def __init__(self, img, detector):
        self.img = img
        self.keypoints, self.descriptors = detector.get_features(img)
        
    def des(self):
        return self.descriptors

    def pts(self):
        return self.keypoints

class Detector:
    NN_K            = 2                 # k of K-nearest-neighbor
    NN_DR           = 0.6               # nearest-neighbor distance ratio

    RANSAC_C        = 0.99              # RANSAC confidence
    RANSAC_D        = 3.0               # RANSAC distance

    HOMOGRAPHY_C    = 0.45              # homography confidence
    HOMOGRAPHY_MI_A = 0.18 * numpy.pi   # homography minimum angle

    def __init__(self):
        # Use ORB as an alternative to SURF and SIFT.
        self.detector = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.targets = []

    def _knn_match(self, tl, tr):
        try:
            matches = self.matcher.knnMatch(tl.des(), tr.des(), Detector.NN_K)
        except:
            return []

        f_nnk   = lambda m: len(m) >= 2
        f_nndr  = lambda m: m[0].distance >= Detector.NN_DR * m[1].distance
        return filter(f_nndr, filter(f_nnk, matches))

    def _symmetric_match(self, tl, tr):
        matches_l = self._knn_match(tl, tr)
        matches_r = self._knn_match(tr, tl)

        matches = []
        for ml in matches_l:
            for mr in matches_r:
                if ml[0].queryIdx == mr[0].trainIdx \
                        and ml[0].trainIdx == mr[0].queryIdx:
                    matches.append(ml[0])

        return matches

    def _extract_points(self, matches, tl, tr):
        pl, pr = [], []
        for m in matches:
            pl.append(tl.pts()[m.queryIdx].pt)
            pr.append(tr.pts()[m.trainIdx].pt)

        return numpy.array(pl), numpy.array(pr)

    def _ransac(self, method, matches, tl, tr):
        if len(matches) < 4:
            return matches

        pl, pr = self._extract_points(matches, tl, tr)

        _, inliers = cv2.findFundamentalMat(pl, pr, method,
            Detector.RANSAC_D, Detector.RANSAC_C)

        good_matches = []
        for m, is_inlier in zip(matches, inliers):
            if is_inlier:
                good_matches.append(m)

        return good_matches

    def _homography_rect(self, matches, tl, tr):
        if len(matches) < 4:
            return None

        pl, pr = self._extract_points(matches, tl, tr)
        h_mat, inliers = cv2.findHomography(pr, pl, cv2.FM_RANSAC)
        confidence = numpy.average(inliers.reshape(1, len(inliers))[0])

        if confidence < Detector.HOMOGRAPHY_C:
            return None
        
        w, h, _ = tr.img.shape
        rect = numpy.array([[(0, 0), (0, w), (h ,w), (h, 0)]], numpy.float32)
        poly = cv2.perspectiveTransform(rect, h_mat).astype(int)

        if not util.is_quadrangle(poly[0]):
            return None

        if min(util.angles(poly[0])) < Detector.HOMOGRAPHY_MI_A:
            return None

        return cv2.boundingRect(poly)

    def _match(self, tl, tr):
        # Reference the RobustMatcher for matching
        # http://stackoverflow.com/a/9894455/881930
        matches = self._symmetric_match(tl, tr)
        matches = self._ransac(cv2.FM_RANSAC, matches, tl, tr)
        return self._homography_rect(matches, tl, tr)

    def add_image(self, img):
        if type(img) == str:
            img = cv2.imread(img)

        self.targets.append(Target(img, self))

    def detect(self, img):
        source = Target(img, self)

        rects = []
        for target in self.targets:
            rect = self._match(source, target)

            if rect is not None:
                rects.append(rect)

        return rects

    def get_features(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.detector.detectAndCompute(gray_img, None)

class Tracker:
    def set_image(self, img):
        pass

    def update_frame(self, frame):
        pass
