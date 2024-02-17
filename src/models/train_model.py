# train_model.py
import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


def train_model(X_train, y_train, X_test, y_test, n_estimators, max_depth, seed):
    # Train your machine learning model

    # Standardize the features (optional but often recommended)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(X_train_scaled, y_train)
    return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    data_path =  input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = 'weekly_sales'
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    #features = train_features[['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
    #target = train_features['TARGET']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_model = train_model(X_train, y_train, X_test, y_test, params['n_estimators'], params['max_depth'], params['seed'])
    save_model(trained_model, output_path)

    

if __name__ == "__main__":
    main()