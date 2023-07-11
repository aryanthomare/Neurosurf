import csv

def calculate_sample_rate(csv_file):
    timestamps = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the timestamp is in the last column
            timestamp = float(row[-1])
            timestamps.append(timestamp)

    # Calculate the time difference between consecutive timestamps
    time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

    # Calculate the average time difference in seconds
    avg_time_diff = sum(time_diffs) / len(time_diffs)

    # Calculate the sample rate in Hz
    sample_rate = 1 / avg_time_diff

    return sample_rate
import os
def list_files(directory):
    files = os.listdir(directory)
    return files

print(list_files("."))

# Provide the path to your CSV file
csv_file_path = "C:\\Users\\aryan\\OneDrive\\Desktop\\PROJECT\\Neurosurf\\AryansOldFiles\\t2.csv"

# Call the function to calculate the sample rate
result = calculate_sample_rate(csv_file_path)

# Print the result
print("Sample rate: {:.2f} Hz".format(result))




