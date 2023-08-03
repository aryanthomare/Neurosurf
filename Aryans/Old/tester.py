import pandas as pd

# Read the input CSV file
input_file = 'Me_Calm.csv'
df = pd.read_csv(input_file, header=None)

# Calculate the sum of each row
row_sums = df.sum(axis=1)

# Divide each value by the sum and create a new DataFrame
normalized_df = df.divide(row_sums, axis=0)

# Write the normalized DataFrame to a new CSV file
output_file = 'output.csv'
normalized_df.to_csv(output_file, index=False, header=False)

print("Normalized data written to", output_file)