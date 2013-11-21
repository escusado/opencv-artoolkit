from termcolor import colored
import sys
import numpy as np
import cv2

cap = None
i = 0

trackers = {
        "rho" : [2, 100, lambda x: handle_change("rho", x)],
        "votes" : [45, 200, lambda x: handle_change("votes", x)],
        "minLength" : [1, 200, lambda x: handle_change("minLength", x)],
        "maxGap" : [1, 200, lambda x: handle_change("maxGap", x)],
        "slopeThreshold" : [50, 200, lambda x: handle_change("slopeThreshold", x)],
        }

def handle_change(key, value):
    global trackers
    if value <= 0:
        value = 1
    trackers[key][0] = value

def define_trackers():
    global trackers
    for tracker in trackers:
        values = trackers[tracker]
        cv2.createTrackbar(tracker, 'frame', values[0], values[1], values[2])

def draw_text(img):
    global trackers
    i = 0

    for tracker in trackers:
        draw_text_line(img, tracker, trackers[tracker][0], (i + 1) * 16)
        i = i + 1

    return img

def draw_text_line(img, key, value, height):
    cv2.putText(img, key+ ": " + str(value), (10, height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def create_window():
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

def load_video():
    "Loads a video with CV"

    global cap, i
    i = 0
    cap = cv2.VideoCapture("checo-test.mp4")

    if cap.isOpened():
        print colored("Video capture opened!", "green")
    else:
        print colored("Video capture failed!", "red")
        exit(1)

def extract_green(img):

    height, width, depth = img.shape

    img_threshold = np.zeros((height, width, 3), np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([40, 120, 120])
    upper_bound = np.array([72, 255, 255])

    img_threshold = cv2.GaussianBlur(img_hsv, (11,11), 0)
    img_threshold = cv2.inRange(img_threshold, lower_bound, upper_bound)
    img_threshold = cv2.bitwise_not(img_threshold)
    img_threshold = cv2.Canny(img_threshold, 0, 200)

    return img_threshold

def track_features(grey_image, original_image):
    global trackers

    img_copy = original_image.copy()

    lines = cv2.HoughLinesP(grey_image, trackers["rho"][0], np.pi / 180,
            trackers["votes"][0], trackers["minLength"][0],
            trackers["maxGap"][0])

    height, width, depth = original_image.shape

    # ZERO THE IMAGE
    #img_copy = np.zeros((height, width, 3), np.uint8)

    if lines is not None:
        #for line in lines[0]:
        #    cv2.line(img_copy, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 1 )

        intersections = compute_intersections(lines[0])
        intersections = find_important_points(grey_image, intersections)
        for point in intersections:
            cv2.circle(img_copy, point, 5, (255, 0, 255))

    return img_copy

def find_intersections(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = a[2]
    y2 = a[3]

    x3 = b[0]
    y3 = b[1]
    x4 = b[2]
    y4 = b[3]

    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    point = (-1, -1)
    if d:
        pointX = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        pointY = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

        point = (pointX, pointY)

    return point

def compute_intersections(lines):
    intersections = []

    for i in range(0, len(lines)):
        for j in range(i+1, len(lines)):
            m1 = line_slope(lines[i])
            m2 = line_slope(lines[j])

            if m1 != m2:
                dm = abs(m2 - m1)
                if dm > trackers["slopeThreshold"][0]:
                    intersections.append(find_intersections(lines[i], lines[j]))

    return intersections

def line_slope(line):
    dx = line[2] - line[0]
    dy = line[3] - line[1]

    if dx != 0:
        return dy / float(dx)
    else:
        return float("inf")

def find_important_points(img, points):
    "From a list of points, extract the most meaningful ones"

    averaged_points = []
    threshold = 200;
    height, width = img.shape

    maxValues = map(max, zip(*points))
    minValues = map(min, zip(*points))


    points = filter(lambda x: x[0] >= -threshold and x[1] >= -threshold
            and x[0] <= width + threshold and x[1] <= height + threshold , points)

    midPoint = get_average_point(points)

    northwest_points = filter(lambda x: x[0] < midPoint[0] and x[1] < midPoint[1], points)
    northeast_points = filter(lambda x: x[0] < midPoint[0] and x[1] > midPoint[1], points)
    southwest_points = filter(lambda x: x[0] > midPoint[0] and x[1] < midPoint[1], points)
    southeast_points = filter(lambda x: x[0] > midPoint[0] and x[1] > midPoint[1], points)

    averaged_points.append(get_average_point(northwest_points))
    averaged_points.append(get_average_point(northeast_points))
    averaged_points.append(get_average_point(southwest_points))
    averaged_points.append(get_average_point(southeast_points))

    return averaged_points


def get_average_point(points):
    averageX = np.mean(map(lambda x: x[0], points))
    averageY = np.mean(map(lambda x: x[1], points))
    return (int(round(averageX)), int(round(averageY)))

def process_frames():
    "Reads the frames from the capture device and processes"

    global cap, i

    while True:
        ret, frame = cap.read()

        if ret:
            img_threshold = extract_green(frame)
            final_frame = track_features(img_threshold, frame)
            #final_frame = img_threshold

            final_frame = resize_frame(final_frame)
            final_frame = draw_text(final_frame)
            cv2.imshow('frame', final_frame)
            cv2.imwrite('points' + str(i) + '.png', final_frame)
            i = i + 1
        else:
            load_video()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

def resize_frame(img):
    return cv2.resize(img, (960, 540))

def cleanup():
    global cap
    cap.release()
    cv2.destroyAllWindows()

def run():
    create_window()
    define_trackers()
    load_video()
    process_frames()

run()
