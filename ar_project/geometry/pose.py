import cv2

#boundaries in white region
def find_object_center(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    #finds largest contour
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cx = x + w // 2
    cy = y + h // 2

    return (cx, cy)