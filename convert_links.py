import csv

# Input and output CSV file paths
input_csv = 'unseen_archaea/ref_genomes/unseen_archaea.csv'
output_csv = 'unseen_archaea/ref_genomes/unseen_archaea_genomes.csv'

# Open the input CSV file and the output CSV file
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    # Create a CSV reader and writer
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Loop through each row in the input CSV
    for row in reader:
        # Extract the filename from the URL (the part after the last '/')
        full_url = row[0]
        filename = full_url.split('/')[-1].replace('.gz', '')  # Remove '.gz'
        
        # Write the extracted filename to the new CSV
        writer.writerow([filename])

print("Extraction complete! Check the output file:", output_csv)
