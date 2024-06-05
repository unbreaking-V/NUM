import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
import sys


# Load the parameters from the params.yaml file
params = yaml.safe_load(open('params.yaml'))['train']
max_depth = params['max_depth']
min_child_weight = params['min_child_weight']
subsample = params['subsample']
colsample_bytree = params['colsample_bytree']
n_estimators = params['n_estimators']
learning_rate = params['learning_rate']
objective = params['objective']
cv = params['cv']
n_jobs = params['n_jobs']
verbose = params['verbose']

# Read paths from the command line
input_dir = sys.argv[1]
output_dir = sys.argv[2]

Path(output_dir).mkdir(exist_ok=True)

# Make the paths
train_file = Path(input_dir) / 'train.csv'
train_target_file = Path(input_dir) / 'train_target.csv'
model_file = Path(output_dir) / 'model.pkl' 

# Load the data
X_train = pd.read_csv(train_file)
y_train = pd.read_csv(train_target_file).squeeze("columns")

# Train the model
xgb1 = XGBRegressor()
parameters = {
    'max_depth':max_depth,
    'min_child_weight':min_child_weight,
    'subsample':subsample,
    'colsample_bytree':colsample_bytree,
    'n_estimators':n_estimators,
    'learning_rate':learning_rate,
    'objective':objective
}

# Create the grid search object
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = cv,
                        n_jobs = n_jobs,
                        verbose=verbose)


# Fit the model
xgb_grid.fit(X_train, y_train)


# Save the model
joblib.dump(xgb_grid.best_estimator_, model_file)