import asyncio

import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
from PIL import Image
from IPython.display import Video
import nb_helpers



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

body_coordinates = {}
async def read_upper_body_coordinates(landmarks):
    global body_coordinates
    body_coordinates["left_shoulder"] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]]
    body_coordinates["right_shoulder"] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
    body_coordinates["left_elbow"] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]]
    body_coordinates["right_elbow"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]]

async def read_lower_body_coordinates(landmarks):
    global body_coordinates
    body_coordinates["left_knee"] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]]
    body_coordinates["right_knee"] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]]
    body_coordinates["left_hip"] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]]
    body_coordinates["right_hip"] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]
    body_coordinates["left_ankle"] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]]
    body_coordinates["right_ankle"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]]


def angle_between_points(p1, p2, p3):
    # Convert points to numpy arrays
   # print(len(p1[0]))


    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    # Create vectors from p1->p2 and p3->p2
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (mag_v1 * mag_v2)

    # Ensure the cosine value is within the range [-1, 1] to avoid errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Get the angle in radians and convert to degrees
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees


cap = cv2.VideoCapture(0)
async def main():
    while cap.isOpened():
        # read frame
        _, frame = cap.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        try:
            # resize the frame for portrait video
            # frame = cv2.resize(frame, (350, 600))
            # convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the frame for pose detection
            pose_results = pose.process(frame_rgb)
            # print(pose_results.pose_landmarks)

            # draw skeleton on the frame
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # display the frame
            # Get coordinates
            try:

                landmarks = pose_results.pose_landmarks.landmark
                await read_upper_body_coordinates(landmarks)
                await read_lower_body_coordinates(landmarks)
                ans = angle_between_points(body_coordinates["left_hip"], body_coordinates["left_shoulder"],
                                           body_coordinates["left_elbow"])
                print(ans)
                cv2.putText(frame, str(ans),
                            (300, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            except:
                pass

            cv2.imshow('Output', frame)

        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    asyncio.run(main())
