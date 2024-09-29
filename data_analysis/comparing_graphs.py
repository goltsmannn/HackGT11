from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np


weighted_csv = pd.read_csv('../weighted.csv')
reference_csv = pd.read_csv('../reference.csv')
body_angles = ['left_hip_angle', 'left_knee_angle', 'right_hip_angle', 'right_knee_angle']

dtw_distances = {}

for body_angle in body_angles:
    weighted_series =  weighted_csv[body_angle].values.reshape(-1, 1)
    reference_series = reference_csv[body_angle].values.reshape(-1, 1)
    # Compute the DTW distance between the two series
    distance, path = fastdtw(weighted_series, reference_series, dist=euclidean)

    # Store the DTW distance
    dtw_distances[body_angle] = [distance]
    print(f'DTW distance for {body_angle}: {distance}')


print(dtw_distances)

df = pd.DataFrame(dtw_distances)
df.to_csv('dtw_distances.csv', mode='a', index=False)
print("Distances written to dtw_distances.csv!")

