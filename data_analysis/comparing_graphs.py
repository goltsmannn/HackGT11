from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

weighted_csv = None
reference_csv = None
dtw_distances = {}


def compare_graphs():
    global weighted_csv, reference_csv, dtw_distances
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    weighted_path = os.path.join(project_root, 'weighted.csv')
    reference_path = os.path.join(project_root, 'reference.csv')
    weighted_csv = pd.read_csv(weighted_path)
    reference_csv = pd.read_csv(reference_path)

    body_angles = weighted_csv.columns[weighted_csv.columns != 'frame_idx'].tolist()

    for body_angle in body_angles:
        weighted_series =  weighted_csv[body_angle].values.reshape(-1, 1)
        reference_series = reference_csv[body_angle].values.reshape(-1, 1)
        distance, path = fastdtw(weighted_series, reference_series, dist=euclidean)
        dtw_distances[body_angle] = [distance]
        print(f'DTW distance for {body_angle}: {distance}')


    print(dtw_distances)

    df = pd.DataFrame(dtw_distances)

    df.to_csv('dtw_distances.csv', mode='a', index=False)
    print("Distances written to dtw_distances.csv!")

def plot_graphs():
    global dtw_distances
    time_weighted = weighted_csv['frame_idx']
    time_reference = reference_csv['frame_idx']

    # Extract angles for weighted and reference CSVs
    hip_angle_weighted = weighted_csv['hip_angle']
    knee_angle_weighted = weighted_csv['knee_angle']
    shin_angle_weighted = weighted_csv['shin_angle']

    hip_angle_reference = reference_csv['hip_angle']
    knee_angle_reference = reference_csv['knee_angle']
    shin_angle_reference = reference_csv['shin_angle']

    plt.figure(figsize=(10, 8))

    # Plot hip_angle for both weighted and reference data
    plt.subplot(3, 1, 1)
    plt.plot(time_weighted, hip_angle_weighted, label='Weighted Hip Angle', color='blue')
    plt.plot(time_reference, hip_angle_reference, label='Reference Hip Angle', color='cyan', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Hip Angle (degrees)')
    plt.title(f'Hip Angle Over Time (Weighted vs Reference, DTW: {dtw_distances['hip_angle']})')
    plt.legend()
    plt.grid(True)

    # Plot knee_angle for both weighted and reference data
    plt.subplot(3, 1, 2)
    plt.plot(time_weighted, knee_angle_weighted, label='Weighted Knee Angle', color='green')
    plt.plot(time_reference, knee_angle_reference, label='Reference Knee Angle', color='lime', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Knee Angle (degrees)')
    plt.title(f'Knee Angle Over Time (Weighted vs Reference), DTW: {dtw_distances['knee_angle']})')
    plt.legend()
    plt.grid(True)

    # Plot shin_angle for both weighted and reference data
    plt.subplot(3, 1, 3)
    plt.plot(time_weighted, shin_angle_weighted, label='Weighted Shin Angle', color='red')
    plt.plot(time_reference, shin_angle_reference, label='Reference Shin Angle', color='orange', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Shin Angle (degrees)')
    plt.title(f'Shin Angle Over Time (Weighted vs Reference), DTW: {dtw_distances['shin_angle']})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def classify_results():
    global dtw_distances
    knee_weight = 0.6
    hip_weight = 0.35
    shin_weight = 0.15
    combined_dtw = dtw_distances['shin_angle'][0] * shin_weight+ dtw_distances['knee_angle'][0] * knee_weight + dtw_distances['hip_angle'][0] * hip_weight
    if combined_dtw <= 1000:
        return "Excellent"
    elif 1000 < combined_dtw <= 1300:
        return "Good"
    elif 1300 < combined_dtw <= 1600:
        return "Average"
    elif 1600 < combined_dtw <= 2000:
        return "Poor"
    else:
        return "Needs Improvement"

def run_post_analysis():
    compare_graphs()
    plot_graphs()
    print(classify_results())