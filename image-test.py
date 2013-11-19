from termcolor import colored
import sys
import numpy as np
import cv2

img = None

def usage():
    "Print the usage for this app."
    sys.stderr.write(str.format('Usage: {0} imagepath', sys.argv[0]))

def check_arguments():
    "Checks proper arguments"

    if len(sys.argv) < 2:
        usage()
        exit(1)


def load_image():
    "Loads an image with CV"

    global img
    img = cv2.imread(sys.argv[1], 0)

    if img is not None:
        print colored("Image loaded succesfully", "green")
    else:
        print colored(str.format("{0} was not found", sys.argv[1]), "red")
        usage()
        exit(1)

def show_image():
    "Creates a window and shows the image"

    global img

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        write_image()
        cv2.destroyAllWindows()

def write_image():
    "Writes the grayscale image to disk"

    global img

    print colored("Image saved ", 'blue'), "as grayscale.png"
    cv2.imwrite('grayscale.png', img)

def run():
    check_arguments()
    load_image()
    show_image()

run()
