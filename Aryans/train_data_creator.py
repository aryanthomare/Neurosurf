import csv
import random
import os

print(os.listdir())

def merge_csv_random_with_indicator(file1, file2, output_file, indicator_file):
    with open(file1, 'r', newline='') as csv_file1, open(file2, 'r', newline='') as csv_file2, open(output_file, 'w', newline='') as output_csv, open(indicator_file, 'w', newline='') as indicator_csv:
        reader1 = list(csv.reader(csv_file1))
        reader2 = list(csv.reader(csv_file2))
        writer = csv.writer(output_csv)
        indicator_writer = csv.writer(indicator_csv)

        # Shuffle lines from both files
        random.shuffle(reader1)
        random.shuffle(reader2)

        # Determine the minimum length of the two readers
        min_length = min(len(reader1), len(reader2))

        # Write randomly picked lines from both files to the output files
        for i in range(min_length):
            # Write to the first output file
            writer.writerow(reader1[i])
            writer.writerow(reader2[i])

            # Write to the second output file with an indicator (0 for file 1, 1 for file 2)
            indicator_writer.writerow([0])
            indicator_writer.writerow([1])

    print(f"Randomly merged data from {file1} and {file2} into {output_file} and {indicator_file}")

# Example usage:
file1_path = 'Aryans/Exported_Values/blinks/blinksTP9.csv'
file2_path = 'Aryans/Exported_Values/normal/normalTP9.csv'
output_file_path = 'Aryans/files_for_training/bn_tp9_test.csv'
indicator_file = 'Aryans/files_for_training/bn_tp9_ans_2.csv'

merge_csv_random_with_indicator(file1_path, file2_path, output_file_path,indicator_file)
