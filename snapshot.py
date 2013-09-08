#!/usr/bin/env python
import cv2

def run(args):
    # Open `VideoCapture` from the first camera if no file specified.
    if args.video_file is not None:
        capture = cv2.VideoCapture(args.video_file)
    else:
        capture = cv2.VideoCapture(0)

    # Read the first frame from `VideoCapture`.
    ret, img = capture.read()
    if ret:
        cv2.imwrite(args.output_file, img)

    capture.release()

if __name__ == '__main__':
    import argparse, os

    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Take a snapshot from a video file or the camera.')
    parser.add_argument('output_file', metavar='O', help='output file name')
    parser.add_argument('-f', '--video_file', default=None,
        help='video file name')

    args = parser.parse_args()

    # Convert to absolute path to get around the path problem on Mac.
    # http://stackoverflow.com/a/9396912/881930
    if args.video_file is not None:
        args.video_file = os.path.join(os.getcwd(), args.video_file)

    run(args)
