#!/usr/bin/env python
import cv2, numpy
import util

class Target:
    def __init__(self, img):
        self.img = cv2.GaussianBlur(numpy.copy(img), (5, 5), 0)
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

    def match(self, tl, tr):
        # Reference the RobustMatcher for matching
        # http://stackoverflow.com/a/9894455/881930
        matches = self._symmetric_match(tl, tr)
        matches = self._ransac(cv2.FM_RANSAC, matches, tl, tr)
        return self._homography_rect(matches, tl, tr)

    def get_features(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.detector.detectAndCompute(gray_img, None)

class Tracker:
    MAX_ITERATIONS  = 50

    def __init__(self, img):
        self.term_criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            Tracker.MAX_ITERATIONS, 1)
        self.track_window = None

        if type(img) == str:
            img = cv2.imread(img)

        self.target = Target(img)

    def clear(self):
        self.track_window = None

    def detect(self, img):
        source = Target(img)
        return detector.match(source, self.target)

    def track(self, img):
        if self.track_window is None:
            return None

        # Reference the OpenCV2 Python Tutorials by abidrahmank for tracking
        # https://github.com/abidrahmank/OpenCV2-Python-Tutorials/ (buggy)
        # and Panaroid Android for bug-fixing
        # http://jayrambhia.wordpress.com/2012/07/11/face-tracking-with-camshift-using-opencvsimplecv/
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)

        ret, window = cv2.meanShift(dst, self.track_window, self.term_criteria)

        if ret < Tracker.MAX_ITERATIONS:
            self.track_window = window
        else:
            self.track_window = None

        return self.track_window

    def update_window(self, img, new_window):
        if self.track_window is not None:
            return False

        self.track_window = new_window

        # Setup ROI for tracking.
        x, y, w, h = self.track_window
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, numpy.array((0., 60., 32.)),
            numpy.array((180., 255., 255.)))

        hsv_roi, mask_roi = hsv[y:y+h, x:x+w], mask[y:y+h, x:x+w]
        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [32], [0, 180])

        self.hist = cv2.normalize(hist, 0, 255, cv2.NORM_MINMAX)
        return True

# Global instance for detector.
detector = Detector()
