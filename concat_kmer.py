import pandas as pd

# Load the frequency vectors from each CSV file without headers
df_2mer = pd.read_csv('/home/chadmi.20/AMAISE_test_human_75_nanopore_test_len_greater_1000_2mer.csv', header=None)
df_3mer = pd.read_csv('/home/chadmi.20/AMAISE_test_human_75_nanopore_test_len_greater_1000_3mer.csv', header=None)

# Concatenate the frequency vectors
combined_df = pd.concat([df_2mer, df_3mer], axis=1)

# Save the combined vectors to a new CSV file
combined_df.to_csv('AMAISE_test_human_75_nanopore_test_len_greater_1000_2mer_3mer.csv', index=False, header=False)
