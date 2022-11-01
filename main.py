import cv2

import numpy as np


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display', image.shape[1], image.shape[0])
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale_17_levels(image):
    high = 255
    while (1):
        low = high - 15
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray, col_to_be_changed_low, col_to_be_changed_high)
        gray[curr_mask > 0] = (high)
        high -= 15
        if (low == 0):
            break

def get_area_of_circle(image_in):
    image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    output = []
    high = 255
    first = True
    while(True):
        low = high - 15
        if(first == False):

            #fkewnfknwelf
            to_be_black_low = np.array([high])
            to_be_black_high = np.array([255])
            curr_mask = cv2.inRange(image, to_be_black_low,
                                    to_be_black_high)
            image[curr_mask > 0] = (0)
        ret, threshold = cv2.threshold(image, low, 255, 0)
        contours, hirechy = cv2.findContours(threshold, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)
        if(len(contours) > 0):
            output.append([cv2.contourArea(contours[0])])
            cv2.drawContours(image_in, contours, -1, (0,0,255), 3)
            high -= 15
            first = False
        if(low == 0):
            break
    return output


image = cv2.imread('image1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(get_area_of_circle(image))
#grayscale_17_levels(gray)
viewImage(gray)