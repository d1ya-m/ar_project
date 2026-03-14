import cv2
import numpy as np

def segment_object(image):
    #convert to gray scale: dark- low val, light-high val
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # >120->255 (white), <=120->0(black)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    return mask