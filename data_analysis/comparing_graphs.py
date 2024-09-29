from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import os
import sys


def compare_graphs():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    weighted_path = os.path.join(project_root, 'weighted.csv')
    reference_path = os.path.join(project_root, 'reference.csv')
    weighted_csv = pd.read_csv(weighted_path)
    reference_csv = pd.read_csv(reference_path)

    body_angles = weighted_csv.columns[weighted_csv.columns != 'frame_idx'].tolist()
    dtw_distances = {}

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

compare_graphs()