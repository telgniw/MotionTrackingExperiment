#!/usr/bin/env python
import cv, cv2, numpy, random, os

class Window:
    def __init__(self, window_name, fps):
        self.name = window_name
        self.fps = fps

        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def draw(self, img):
        cv2.imshow(self.name, img)

        # Wait for 1000/fps milliseconds to prevent window from blocking.
        cv2.waitKey(int(1000 / self.fps));

class Tracker:
    def __init__(self, video_file):
        # Convert to absolute path to get around the problem on Mac.
        # http://stackoverflow.com/a/9396912/881930
        video_file = os.path.join(os.getcwd(), video_file)

        self.capture = cv2.VideoCapture(video_file)

        # Get video information.
        self.length = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
        self.fps    = self.capture.get(cv.CV_CAP_PROP_FPS)
        self.width  = self.capture.get(cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

        # Create display window.
        self.window = Window('Motion Tracking Experiment', self.fps)

    def run(self):
        # Default values for BackgroundSubtractorMOG:
        # history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=15.0
        bgsub = cv2.BackgroundSubtractorMOG()

        for i in range(self.length):
            ret, img = self.capture.read()

            if not ret:
                break

            # Get foreground image.
            fg_mask = bgsub.apply(img, learningRate=0.01)
            fg_mask /= 255
            fg_img = numpy.dstack([img[:,:,i] * fg_mask for i in range(3)])

            # Convert to gray scale and threshold to binary image.
            gray_img = cv2.cvtColor(fg_img, cv2.COLOR_RGB2GRAY)
            _, gray_img = cv2.threshold(gray_img, 2, 255, cv2.THRESH_BINARY)
            gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
            _, gray_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)

            # Calculate contours in the image.
            # Make a copy because `findContours` will modify the source image.
            cpy = numpy.copy(gray_img)
            contours, hierarchy = cv2.findContours(cpy, cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)

            if hierarchy is not None:
                idx, color = 0, []
                while idx >= 0:
                    while idx >= len(color):
                        c = [random.randint(0, 99), random.randint(100, 199), random.randint(200, 255)]
                        random.shuffle(c)
                        color.append(tuple(c))
                    cv2.drawContours(img, contours, idx, color[idx], thickness=-1)
                    idx = hierarchy[0][idx][0]

            self.window.draw(img)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Motion tracking.')
    parser.add_argument('video_file', metavar='F', default=None,
        help='video file name')

    args = parser.parse_args()

    # Run motion tracking.
    tracker = Tracker(args.video_file)
    tracker.run()
