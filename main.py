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
    high = 200
    first = True
    while(True):
        low = high - 200
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
            cv2.drawContours(image_in, contours, -1, (0,0,200), 3)
            high -= 10
            first = False
        if(low <= 0):
            break
    return output


green = np.uint8([[[0, 255, 0]]])
image = cv2.imread('image2.png')
#print(get_area_of_circle(image))
#grayscale_17_levels(gray)
viewImage(image)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img) #view1
green_low = np.array([45, 100, 50])
green_high = np.array([75, 255, 255])
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
hsv_img[curr_mask > 0] = ([75, 255, 200])
viewImage(hsv_img) #view2

image_gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB),
                          cv2.COLOR_RGB2GRAY)
viewImage(image_gray) #view3

ret, threshold = cv2.threshold(image_gray, 90, 255, 0)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

main_contours =max ([[cv2.contourArea(contours[i]), i] for i in range(0, len(contours)-1)])

print(main_contours)

cv2.drawContours(image, contours, main_contours[1], (0, 0, 255), 3)
viewImage(image)