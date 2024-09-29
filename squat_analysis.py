import asyncio
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

body_coordinates = {}
reference_data = []
weighted_data = []
squat_in_progress = False

# Required landmarks for squat analysis
required_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

def angle_between_points(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (mag_v1 * mag_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees

def all_landmarks_visible(landmarks):
    """Check if all required landmarks are visible."""
    if landmarks is None:
        return False
    for lm in required_landmarks:
        if landmarks[lm.value].visibility < 0.5:  # Check visibility score
            return False
    return True

async def read_body_coordinates(landmarks):
    global body_coordinates
    body_coordinates["left_shoulder"] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    body_coordinates["right_shoulder"] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    body_coordinates["left_hip"] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    body_coordinates["right_hip"] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    body_coordinates["left_knee"] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    body_coordinates["right_knee"] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    body_coordinates["left_ankle"] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
    body_coordinates["right_ankle"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    body_coordinates["left_foot_index"] = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    body_coordinates["right_foot_index"] = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

def detect_squat_start():
    left_hip_angle = angle_between_points(body_coordinates["left_shoulder"], body_coordinates["left_hip"], body_coordinates["left_knee"])
    return left_hip_angle < 140  # Customize the threshold as needed

def record_angles():
    angles = {
        'left_hip_angle': angle_between_points(body_coordinates["left_shoulder"], body_coordinates["left_hip"], body_coordinates["left_knee"]),
        'right_hip_angle': angle_between_points(body_coordinates["right_shoulder"], body_coordinates["right_hip"], body_coordinates["right_knee"]),
        'left_knee_angle': angle_between_points(body_coordinates["left_hip"], body_coordinates["left_knee"], body_coordinates["left_ankle"]),
        'right_knee_angle': angle_between_points(body_coordinates["right_hip"], body_coordinates["right_knee"], body_coordinates["right_ankle"]),
        'left_ankle_angle': angle_between_points(body_coordinates["left_knee"], body_coordinates["left_ankle"], body_coordinates["left_foot_index"]),
        'right_ankle_angle': angle_between_points(body_coordinates["right_knee"], body_coordinates["right_ankle"], body_coordinates["right_foot_index"])
    }
    return angles

cap = cv2.VideoCapture(0)

async def main(mode='reference'):
    global squat_in_progress, reference_data, weighted_data

    while cap.isOpened():
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
            
            # Check if all required landmarks are visible
            if not all_landmarks_visible(landmarks):
                print("Some landmarks are not visible; skipping frame.")
                continue

            await read_body_coordinates(landmarks)

            if detect_squat_start() and not squat_in_progress:
                squat_in_progress = True
                if mode == 'reference':
                    reference_data = []
                else:
                    weighted_data = []
                print("Squat started")

            if squat_in_progress:
                angles = record_angles()
                if mode == 'reference':
                    reference_data.append(angles)
                else:
                    weighted_data.append(angles)

                if not detect_squat_start():
                    squat_in_progress = False
                    print("Squat ended")
        
        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Squat Analysis', frame)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)