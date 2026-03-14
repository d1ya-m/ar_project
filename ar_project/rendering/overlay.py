import cv2

def overlay_cube(image, center):

    x,y = center

    cv2.rectangle(
        image,
        (x-50,y-50),
        (x+50,y+50),
        (255,0,0),
        3
    )

    return image