import sys
import json
import yaml
import pickle
import pandas as pd
from pathlib import Path
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error


# Read the command-line arguments
model_dir = sys.argv[1]
input_dir = sys.argv[2]
scores_file = sys.argv[3]


# Make the paths
train_file = Path(input_dir) / 'train.csv'
train_target_file = Path(input_dir) / 'train_target.csv'
test_file = Path(input_dir) / 'test.csv'
test_target_file = Path(input_dir) / 'test_target.csv'
val_file = Path(input_dir) / 'val.csv'
val_target_file = Path(input_dir) / 'val_target.csv'
model_file = Path(model_dir) / 'model.pkl'

# Load the model
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Load the data
X_test = pd.read_csv(test_file)
y_test = pd.read_csv(test_target_file).squeeze("columns")
X_train = pd.read_csv(train_file)
y_train = pd.read_csv(train_target_file).squeeze("columns")
X_val = pd.read_csv(val_file)
y_val = pd.read_csv(val_target_file).squeeze("columns")

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    'mean_absolute_error': mae,
    'mean_squared_error': mse,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

# Save metrics to a file
with open(scores_file, 'w') as f:
    json.dump(metrics, f, indent=4)

