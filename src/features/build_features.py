
import pathlib
import sys
import yaml
import geopy.distance
import joblib
import pandas as pd
import calendar
from sklearn.model_selection import train_test_split

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def Sesson_getter(quarter):
    if quarter ==1:
        return 'Winter'
    elif quarter == 2:
        return 'Spring'
    elif quarter == 3:
        return 'Summer'
    else:
        return 'Fall'

def feature_eng(data):
    # Load your dataset from a given path
    pd.options.display.float_format = '{:,.0f}'.format

    data.columns = data.columns.str.lower()
    # Correcting the date format
    data['date'] = pd.to_datetime(data['date'], format = "%d-%m-%Y")

    data['year'] = data['date'].dt.year

    data['quarter'] = data['date'].dt.quarter
    
    data['season'] = data['quarter'].apply(Sesson_getter)

    data['month'] = data['date'].dt.month

    data['month_name'] = data['date'].dt.month_name()

    data['week'] = data['date'].dt.isocalendar().week

    data['day_of_week'] = data['date'].dt.day_name()

    # remove categorial colunm to numerical value using one hot encoding
    final_data = pd.get_dummies(data, columns=['season', 'month'])

    # drop day_of_week and month_name
    final_data = final_data.drop(['day_of_week', 'month_name', 'date'], axis=1)

    return final_data

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = sys.argv[1]
    data_path =  input_file
    output_path = home_dir.as_posix() + '/data/processed'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    df = feature_eng(data)
    
    train_data, test_data = split_data(df, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

    

if __name__ == "__main__":
    main()