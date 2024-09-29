import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_squat_data(filename):
    """Load squat data from a CSV file."""
    squat_data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            squat_data.append({
                'frame_idx': int(row['frame_idx']),
                'left_hip_angle': float(row['left_hip_angle']),
                'right_hip_angle': float(row['right_hip_angle']),
                # Add more angles here as needed
            })
    return squat_data

def average_reference_data(reference_data_list):
    """Average data across multiple reference squats."""
    num_frames = len(reference_data_list[0])
    averaged_data = []
    for i in range(num_frames):
        frame_data = {key: 0 for key in reference_data_list[0][i].keys()}
        for data in reference_data_list:
            for key in data[i]:
                frame_data[key] += data[i][key]
        averaged_data.append({key: value / len(reference_data_list) for key, value in frame_data.items()})
    return averaged_data

def interpolate_data(data, target_length):
    """Interpolate data to match the target length."""
    interpolated_data = []
    num_frames = len(data)
    x_original = np.linspace(0, 1, num_frames)
    x_target = np.linspace(0, 1, target_length)

    for key in data[0]:
        values = [frame[key] for frame in data]
        interpolation_func = interp1d(x_original, values, kind='linear')
        interpolated_values = interpolation_func(x_target)

        # Append interpolated data to the list
        for i in range(target_length):
            if i >= len(interpolated_data):
                interpolated_data.append({})
            interpolated_data[i][key] = interpolated_values[i]
    
    return interpolated_data

def compute_deviation(reference_data, rep_data):
    """Compute the deviation between reference and a squat rep."""
    deviation = 0
    for i in range(len(reference_data)):
        for key in reference_data[i]:
            deviation += abs(reference_data[i][key] - rep_data[i][key])
    return deviation

def analyze_squat_set(reference_filename, weighted_filename):
    # Load reference and weighted data
    reference_data_list = load_squat_data(reference_filename)
    weighted_data_list = load_squat_data(weighted_filename)

    # Average reference data
    averaged_reference_data = average_reference_data(reference_data_list)

    # Interpolate weighted data to match the length of the averaged reference data
    target_length = len(averaged_reference_data)
    interpolated_weighted_data = interpolate_data(weighted_data_list, target_length)

    # Compute deviation for each rep
    deviations = [compute_deviation(averaged_reference_data, interpolate_data([rep], target_length)) for rep in interpolated_weighted_data]

    # Sort reps by deviation
    sorted_reps = sorted(zip(deviations, interpolated_weighted_data), key=lambda x: x[0], reverse=True)

    # Visualize results
    for i, (deviation, rep_data) in enumerate(sorted_reps):
        print(f"Rep {i + 1} deviation: {deviation}")
        plot_angle_data(averaged_reference_data, rep_data)

def plot_angle_data(reference_data, rep_data):
    """Plot angle data for comparison."""
    timestamps = range(len(reference_data))
    plt.figure()
    plt.plot(timestamps, [frame['left_hip_angle'] for frame in reference_data], label='Reference Left Hip')
    plt.plot(timestamps, [frame['left_hip_angle'] for frame in rep_data], label='Rep Left Hip')
    # Add more plots for other angles if needed
    plt.legend()
    plt.show()