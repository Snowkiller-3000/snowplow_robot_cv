import sys
import os
import cv2 as cv
import numpy as np

from ColorDictHSV import color_dict_HSV


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print('Not enough parameters')
        print('Usage:\nmorph_lines_detection.py < path_to_image_dir >')
        return -1

    img_folder = argv[0]
    for filename in os.listdir(img_folder):
        # load image
        img_path = img_folder + filename
        src = cv.imread(img_path, cv.IMREAD_COLOR)

        # Check if image is loaded fine
        if src is None:
            print('Error opening image: ' + img_path)
            continue

        # Show source image
        cv.imshow("src", src)
        show_wait_destroy("src", src)

        # find pylons
        detect_markers(src)

        # draw path



LOW = 1
HIGH = 0


def detect_markers(img):
    # convert image to hsv-space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # we only care about red-ish pixels
    mask_red1 = cv.inRange(hsv, np.array(color_dict_HSV["red1"][LOW]), np.array(color_dict_HSV["red1"][HIGH]))
    mask_red2 = cv.inRange(hsv, np.array(color_dict_HSV["red2"][LOW]), np.array(color_dict_HSV["red2"][HIGH]))
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    reds = cv.bitwise_and(img, img, mask=mask_red)

    # check edges
    edges = cv.Canny(reds, 200, 400)
    show_wait_destroy("mask", mask_red)
    show_wait_destroy("edges", edges)
    final_image = edges
    return final_image


if __name__ == "__main__":
    main(sys.argv[1:])
