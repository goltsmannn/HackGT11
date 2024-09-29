import numpy as np
from enum import Enum

BODY_PARTS = [
    "SHOULDER", "HIP", "KNEE", "ANKLE", "FOOT_INDEX"
]

class Visibility(Enum):
    LEFT = (1, 'left')
    RIGHT = (2, 'right')
    BOTH = (3, 'both')
    NONE = (0, False)

def angle_between_points(p1, p2, p3):
    p1 = np.array(p1)[:2]
    p2 = np.array(p2)[:2]
    p3 = np.array(p3)[:2]

    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
