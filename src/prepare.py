import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import sys

#Read the parameters from the params.yaml file
params = yaml.safe_load(open('params.yaml'))['prepare']
split = params['split']
seed = params['seed']

# Read the input data
input_file = Path(sys.argv[1])
diamond_df = pd.read_csv(input_file, index_col=0)


# Define the output paths
X_train_output = Path('data') / 'prepared' / 'train.csv'
X_val_output = Path('data') / 'prepared' / 'val.csv'
X_test_output = Path('data') / 'prepared' / 'test.csv'

y_train_output = Path('data') / 'prepared' / 'train_target.csv'
y_val_output = Path('data') / 'prepared' / 'val_target.csv'
y_test_output = Path('data') / 'prepared' / 'test_target.csv'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

# Encode the ordinal categorical variable 'cut'
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
diamond_df.cut = diamond_df.cut.map(cut_mapping)

# Encoding the ordinal categorical variable 'color'
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
diamond_df.color = diamond_df.color.map(color_mapping)

# Encoding the ordinal cateogircal variable 'clarity'
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
diamond_df.clarity = diamond_df.clarity.map(clarity_mapping)

# Drop the rows with missing values
diamond_df = diamond_df.drop(diamond_df[diamond_df["x"]==0].index)
diamond_df = diamond_df.drop(diamond_df[diamond_df["y"]==0].index)
diamond_df = diamond_df.drop(diamond_df[diamond_df["z"]==0].index)
diamond_df = diamond_df[diamond_df['depth'] < diamond_df['depth'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['table'] < diamond_df['table'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['x'] < diamond_df['x'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['y'] < diamond_df['y'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['z'] < diamond_df['z'].quantile(0.99)]

# Create a copy of the data
model_df = diamond_df.copy()

# Split the data into features and target
X = model_df.drop(['price'], axis=1)
y = model_df['price']

# Split the data into train, validation and test sets
X_train_val, X_test, y_train_val, y_test =  train_test_split(X, y, test_size=split, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=split, random_state=seed)

#Save the train, validation and test data
X_train.to_csv(X_train_output, index=False)
X_val.to_csv(X_val_output, index=False)
X_test.to_csv(X_test_output, index=False)
y_train.to_csv(y_train_output, index=False, header=True)
y_val.to_csv(y_val_output, index=False, header=True)
y_test.to_csv(y_test_output, index=False, header=True)

