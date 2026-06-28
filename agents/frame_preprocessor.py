import cv2
import numpy as np

def enhance_frame(frame):

    frame = cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        10,
        10,
        7,
        21
    )

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    frame = cv2.filter2D(
        frame,
        -1,
        kernel
    )

    return frame