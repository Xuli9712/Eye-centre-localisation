import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Original CSV filename
filename = r'F:\gi4e_database\image_labels.csv'

# Load the CSV file
df = pd.read_csv(filename)

# Split the dataset into train+validation set and test set
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

# Split the train+validation set into train set and validation set
train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42)

# Get the base filename without extension
base_filename = os.path.splitext(filename)[0]

# Save the datasets to new CSV files
train_df.to_csv(f'{base_filename}_train.csv', index=False)
val_df.to_csv(f'{base_filename}_val.csv', index=False)
test_df.to_csv(f'{base_filename}_test.csv', index=False)
