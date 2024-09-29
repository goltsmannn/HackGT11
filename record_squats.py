import cv2
import mediapipe as mp
import numpy as np
import asyncio
import csv
from scipy.interpolate import interp1d

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper function to compute angles
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
    return np.degrees(angle)

# Detect if a squat has started based on the hip angle
def detect_squat_start(body_coordinates):
    left_hip_angle = angle_between_points(
        body_coordinates["left_shoulder"], body_coordinates["left_hip"], body_coordinates["left_knee"]
    )
    return left_hip_angle < 140  # Adjust threshold as needed

# Extract necessary body coordinates for angle calculations
def read_body_coordinates(landmarks):
    body_coordinates = {
        "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
        "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
        "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
        "right_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
        "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
        "right_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z],
        "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],
        "right_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    }
    return body_coordinates

# Record a set of squats and save to CSV
async def record_squat_set(filename, mode='reference', num_reps=3):
    cap = cv2.VideoCapture(0)
    squat_in_progress = False
    squat_set_data = []
    rep_count = 0

    while cap.isOpened():
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
            if landmarks:
                body_coordinates = read_body_coordinates(landmarks)

                # Collect angles and store them
                angles = {
                    'left_hip_angle': angle_between_points(body_coordinates["left_shoulder"], body_coordinates["left_hip"], body_coordinates["left_knee"]),
                    'right_hip_angle': angle_between_points(body_coordinates["right_shoulder"], body_coordinates["right_hip"], body_coordinates["right_knee"]),
                    'left_knee_angle': angle_between_points(body_coordinates["left_hip"], body_coordinates["left_knee"], body_coordinates["left_ankle"]),
                    'right_knee_angle': angle_between_points(body_coordinates["right_hip"], body_coordinates["right_knee"], body_coordinates["right_ankle"]),
                    # Add more angles if needed
                }

                if detect_squat_start(body_coordinates) and not squat_in_progress:
                    squat_in_progress = True
                    rep_data = []  # Store frames of the current rep
                    print(f"Squat {rep_count + 1} started.")

                if squat_in_progress:
                    rep_data.append(angles)

                if not detect_squat_start(body_coordinates) and squat_in_progress:
                    squat_in_progress = False
                    squat_set_data.append(rep_data)
                    rep_count += 1
                    print(f"Squat {rep_count} ended.")
                    if rep_count >= num_reps:
                        break

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Squat Recording', frame)

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        headers = ['frame_idx', 'left_hip_angle', 'right_hip_angle', 'left_knee_angle', 'right_knee_angle']
        writer.writerow(headers)
        for rep_idx, rep_data in enumerate(squat_set_data):
            for frame_idx, angles in enumerate(rep_data):
                writer.writerow([frame_idx, angles['left_hip_angle'], angles['right_hip_angle'],
                                 angles['left_knee_angle'], angles['right_knee_angle']])
