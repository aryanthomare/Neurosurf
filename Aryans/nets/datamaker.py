import csv
import os
# List of CSV filenames to concatenate
filenamesb = ['Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksAF7.csv',
             'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksAF8.csv', 
             'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksTP9.csv', 
             'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksTP10.csv']
             
filenamesn = ['Neurosurf\\Aryans\\Exported_Values\\normal\\normalAF7.csv',
             'Neurosurf\\Aryans\\Exported_Values\\normal\\normalAF8.csv', 
             'Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP9.csv', 
             'Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP10.csv']

# Open all the files for reading
files = [open(filename, 'r') for filename in filenamesn]# <--- Change this to `filenamesb` to concatenate the blink files
readers = [csv.reader(file) for file in files]

# Create a new file for the concatenated output
with open('Neurosurf\\Aryans\\nets\\outputn.csv', 'w', newline='') as outfile:# <--- Change this to `outputb.csv` to concatenate the blink files
    writer = csv.writer(outfile)

    # Use `zip` to read lines from all files in parallel
    for rows in zip(*readers):
        # Concatenate corresponding rows into one
        concatenated_row = []
        for row in rows:
            concatenated_row.extend(row)
        
        # Write the concatenated row to the output CSV
        writer.writerow(concatenated_row)

# Close the input files
for file in files:
    file.close()