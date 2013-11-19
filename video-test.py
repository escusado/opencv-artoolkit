from termcolor import colored
import sys
import numpy as np
import cv2

cap = None

def load_video():
    "Loads a video with CV"

    global cap
    cap = cv2.VideoCapture("test.mov")

    if cap.isOpened():
        print colored("Video capture opened!", "green")
    else:
        print colored("Video capture failed!", "red")
        exit(1)

def extract_green(img):

    height, width, depth = img.shape

    img_threshold = np.zeros((height, width, 3), np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([58, 100, 100])
    upper_bound = np.array([68, 255, 255])

    img_threshold = cv2.inRange(img_hsv, lower_bound, upper_bound)

    return img_threshold

def track_features(grey_image, original_image):

    img_copy = original_image.copy()

    corners = cv2.goodFeaturesToTrack(grey_image, 4, 0.01, 10)

    for point in corners:

        cv2.circle(img_copy, (point[0][0], point[0][1]), 4, (255, 0, 0), -1 )

    return img_copy


def process_frames():
    "Reads the frames from the capture device and processes"

    global cap

    while True:
        ret, frame = cap.read()

        if ret:
            img_threshold = extract_green(frame)
            cv2.imshow('frame', track_features(img_threshold, frame))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

def cleanup():
    global cap
    cap.release()
    cv2.destroyAllWindows()

def run():
    load_video()
    process_frames()

run()
