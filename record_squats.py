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
    left_visible = True
    right_visible = True
    sum_left_visibility = 0
    sum_right_visibility = 0
    if landmarks is None:
        return False

    for lm in required_left_landmarks:
        sum_left_visibility += landmarks[lm.value].visibility
        if landmarks[lm.value].visibility < 0.3:  # Lower visibility threshold
            left_visible = False

    for lm in required_right_landmarks:
        sum_right_visibility += landmarks[lm.value].visibility
        if landmarks[lm.value].visibility < 0.3:  # Lower visibility threshold
            right_visible = False

    if left_visible and right_visible:
        if(sum_left_visibility > sum_right_visibility):
            return Visibility.LEFT.value
        return Visibility.RIGHT.value
    elif left_visible:
        return Visibility.LEFT.value
    elif right_visible:
        return Visibility.RIGHT.value
    else:
        return Visibility.NONE.value

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
    buffer_frames = 10  # Number of extra frames to capture after detecting squat end

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
            frame_data = {'hip_angle': None, 'knee_angle': None, 'shin_angle': None}  # Initialize frame data with None

            if landmarks:
                squat_flag, squat_side = all_landmarks_visible(landmarks)
                body_coordinates = read_body_coordinates(landmarks)

                # Determine which side to use for calculations based on visibility
                if squat_side:
                    try:
                        frame_data['hip_angle'] = angle_between_points(
                            body_coordinates[f"{squat_side}_shoulder"], body_coordinates[f"{squat_side}_hip"], body_coordinates[f"{squat_side}_knee"]
                        )
                    except KeyError:
                        frame_data['hip_angle'] = None

                    try:
                        frame_data['knee_angle'] = angle_between_points(
                            body_coordinates[f"{squat_side}_hip"], body_coordinates[f"{squat_side}_knee"], body_coordinates[f"{squat_side}_ankle"]
                        )
                    except KeyError:
                        frame_data['knee_angle'] = None

                    try:
                        frame_data['shin_angle'] = angle_between_points(
                            body_coordinates[f"{squat_side}_knee"], body_coordinates[f"{squat_side}_heel"], body_coordinates[f"{squat_side}_foot_index"]
                        )
                    except KeyError:
                        frame_data['shin_angle'] = None

                # Display angles even if some landmarks are missing
                h, w, _ = frame.shape
                if squat_side:
                    hip_pos = (int(body_coordinates[f"{squat_side}_hip"][0] * w), int(body_coordinates[f"{squat_side}_hip"][1] * h))
                    hip_text = f'Hip: {frame_data["hip_angle"]:.2f}' if frame_data["hip_angle"] is not None else "Hip: N/A"
                    cv2.putText(frame, hip_text, hip_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    knee_pos = (int(body_coordinates[f"{squat_side}_knee"][0] * w), int(body_coordinates[f"{squat_side}_knee"][1] * h))
                    knee_text = f'Knee: {frame_data["knee_angle"]:.2f}' if frame_data["knee_angle"] is not None else "Knee: N/A"
                    cv2.putText(frame, knee_text, knee_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    heel_pos = (int(body_coordinates[f"{squat_side}_heel"][0] * w), int(body_coordinates[f"{squat_side}_heel"][1] * h))
                    heel_text = f'Shin: {frame_data["shin_angle"]:.2f}' if frame_data["shin_angle"] is not None else "Shin: N/A"
                    cv2.putText(frame, heel_text, heel_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.putText(frame, f'Squat side: {squat_side}', (10, 30),  # Adjusted coordinates
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # Start/stop recording based on squat detection
                if squat_side and frame_data['hip_angle'] and frame_data['hip_angle'] < 150 and not squat_in_progress:
                    squat_in_progress = True
                    rep_data = []
                    print(f"Squat {rep_count + 1} started.")

                if squat_in_progress:
                    rep_data.append(frame_data)

                if squat_side and frame_data['hip_angle'] and frame_data['hip_angle'] >= 150 and squat_in_progress:
                    if buffer_frames > 0:
                        buffer_frames -= 1
                    else:
                        squat_in_progress = False
                        squat_set_data.append(rep_data)
                        rep_count += 1
                        buffer_frames = 10  # Reset buffer frames for the next squat
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
        headers = ['frame_idx', 'hip_angle', 'knee_angle', 'shin_angle']
        writer.writerow(headers)
        for rep_idx, rep_data in enumerate(squat_set_data):
            for frame_idx, angles in enumerate(rep_data):
                writer.writerow([frame_idx, angles['hip_angle'], angles['knee_angle'], angles['shin_angle']])