import cv2
import mediapipe as mp
import numpy as np
import csv
from utils import angle_between_points, Visibility

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
squat_side = None

required_left_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
]

required_right_landmarks = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]


def all_landmarks_visible(landmarks):
    """Check if all required landmarks are visible."""
    leftSide = True
    rightSide = True
    if landmarks is None:
        return False

    for lm in required_left_landmarks:
        if landmarks[lm.value].visibility < 0.5:  # Check visibility score
            leftSide = False

    for lm in required_right_landmarks:
        if landmarks[lm.value].visibility < 0.5:  # Check visibility score
            rightSide = False

    if rightSide and leftSide:
        return Visibility.LEFT.value
    elif leftSide:
        return Visibility.LEFT.value
    elif rightSide:
        return Visibility.RIGHT.value
    else:
        return Visibility.NONE.value

def detect_squat_start(body_coordinates, squat_side):
    if not squat_side:
        return False
    hip_angle = angle_between_points(
        body_coordinates[f"{squat_side}_shoulder"], body_coordinates[f"{squat_side}_hip"], body_coordinates[f"{squat_side}_knee"]
    )
    
    response =  hip_angle > 140
    return response

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
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z],
        "left_heel": [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z],
        "right_heel": [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z],
        "left_foot_index": [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z],
        "right_foot_index": [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    }
    return body_coordinates

def record_squat_set(filename, num_reps=2):
    global squat_side
    cap = cv2.VideoCapture(0)
    squat_in_progress = False
    squat_set_data = []
    rep_count = 0

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            landmarks = pose_results.pose_landmarks.landmark
            # print(landmarks)
            if landmarks:
                squat_flag, squat_side = all_landmarks_visible(landmarks) 
                if squat_side:
                    print(squat_side)

                body_coordinates = read_body_coordinates(landmarks)

                # Calculate angles, even if some landmarks might be missing
                try:
                    angles = {
                        'left_hip_angle': angle_between_points(body_coordinates["left_shoulder"], body_coordinates["left_hip"], body_coordinates["left_knee"]),
                        'right_hip_angle': angle_between_points(body_coordinates["right_shoulder"], body_coordinates["right_hip"], body_coordinates["right_knee"]),
                        'left_knee_angle': angle_between_points(body_coordinates["left_hip"], body_coordinates["left_knee"], body_coordinates["left_ankle"]),
                        'right_knee_angle': angle_between_points(body_coordinates["right_hip"], body_coordinates["right_knee"], body_coordinates["right_ankle"]),
                        'left_shin_angle': angle_between_points(body_coordinates["left_knee"], body_coordinates["left_heel"], body_coordinates["left_foot_index"]),
                        'right_shin_angle': angle_between_points(body_coordinates["right_knee"], body_coordinates["right_heel"], body_coordinates["right_foot_index"])
                    }
                except KeyError as e:
                    print(f"Error: Missing landmark data for {e}")
                    continue

                h, w, _ = frame.shape

                # Display angles even if some landmarks are missing
                if squat_flag > 0:
                    squat_side_caps = squat_side.upper()

                    hip_pos = (int(body_coordinates[f"{squat_side}_hip"][0] * w), int(body_coordinates[f"{squat_side}_hip"][1] * h))
                    cv2.putText(frame, f'Hip: {angles[f"{squat_side}_hip_angle"]:.2f}', hip_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    knee_pos = (int(body_coordinates[f"{squat_side}_knee"][0] * w), int(body_coordinates[f"{squat_side}_knee"][1] * h))
                    cv2.putText(frame, f'Knee: {angles[f"{squat_side}_knee_angle"]:.2f}', knee_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    foot_pos = (int(body_coordinates[f"{squat_side}_foot_index"][0] * w), int(body_coordinates[f"{squat_side}_foot_index"][1] * h))
                    cv2.putText(frame, f'Shin: {angles[f"{squat_side}_shin_angle"]:.2f}', foot_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.putText(frame, f'Squat side: {squat_side}', (10, 30),  # Adjusted coordinates
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


                # Only start recording if all required landmarks are visible
                rep_data = []

                if squat_flag > 0 and detect_squat_start(body_coordinates, squat_side) and not squat_in_progress:
                    squat_in_progress = True
                    print(f"Squat {rep_count + 1} started.")

                if squat_in_progress:
                    rep_data.append(angles)

                if not detect_squat_start(body_coordinates, squat_side) and squat_in_progress:
                    squat_in_progress = False
                    squat_set_data.append(rep_data)
                    rep_count += 1
                    print(f"Squat {rep_count} ended.")
                    if rep_count >= num_reps:
                        break

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Squat Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['frame_idx', f'{squat_side}_hip_angle',  f'{squat_side}_knee_angle', f'{squat_side}_shin_angle']
        writer.writerow(headers)
        for rep_idx, rep_data in enumerate(squat_set_data):
            for frame_idx, angles in enumerate(rep_data):
                writer.writerow([frame_idx,  
                                 angles[ f'{squat_side}_hip_angle'],
                                 angles[ f'{squat_side}_knee_angle'],
                                 angles[ f'{squat_side}_shin_angle']])
