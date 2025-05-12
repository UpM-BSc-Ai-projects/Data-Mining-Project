import kagglehub
import shutil
import os

import numpy as np 
import pandas as pd

# Data Preprocessing

# Downloading the dataset
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
print("Path to dataset files:", path)

# Copying the dataset to the project directory
# Source and destination paths
source_path = os.path.join(path, 'btcusd_1-min_data.csv')
destination_path = './data/btcusd_dataset.csv'

# Create destination directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Only copy if the file doesn't already exist
if not os.path.exists(destination_path):
    shutil.copy(source_path, destination_path)
    print("File copied successfully.")

else:
    print("File already exists. Skipping copy.")


#  Reading dataset
data = pd.read_csv(destination_path)
data.head()

# Info about dataset
print('Shape of the dataset: ', data.shape)
print(f"info of the dataset: \n{data.info()}\n")
print(f"describe of the dataset: \n{data.describe()}\n")
print(f"null values of the dataset: \n{data.isnull().sum()}\n")
print(f"duplicated values of the dataset: \n{data.duplicated().sum()}\n")

# Cleaning Dataset
# Drop the 'datetime' column
data = data.drop(columns=['datetime'])

# view the updated DataFrame
print(data.head())
print(f"\n\nnull values of the dataset: \n{data.isnull().sum()}\n")

# Convert Unix timestamp (seconds since 00:00:00 UTC January 1, 1970) to datetime
data['datetime'] = pd.to_datetime(data['Timestamp'], unit='s')
print(data.head())

# ensure the data is continuous and their are no missing values or rows,
# Reindexes the data to have a row for every minute — even if that minute was missing in the original data.
continuous_data = data.set_index('datetime').asfreq('min')
print('data Null/NA Values before fill:', continuous_data.isnull().values.sum())

# fill in and interpolate missing values after re-indexing is done
continuous_data.interpolate(method='time', inplace=True)  # Time-based interpolation
continuous_data.ffill(inplace=True) # forwards fill missing values
continuous_data.reset_index(inplace=True) # Moves 'datetime' back from the index to a regular column
print('data Null/NA Values after fill:', continuous_data.isnull().values.sum())

data = continuous_data.copy()

first_nonzero_row = data[data['Volume'] > 0].head(1)
print(first_nonzero_row)

# Save cleaned data to csv file
data.to_csv('./data/cleaned_data.csv', index=False)

# Data Reduction
data = data.drop(columns=['Volume'])
data.head()

def create_resampled_dataframe(data, resample_rule='h'):
    """
    Create a resampled DataFrame for a given time resolution.
        
    Args:
        data (pd.DataFrame): Input DataFrame
        resample_rule (str): Resampling rule (e.g., 'h', 'D', 'W-MON', 'M')
        
    Returns:
        pd.DataFrame: Resampled DataFrame
        
    """
    datetime_column = data['datetime']

    if resample_rule == '1min' or resample_rule is None:
        return data

    df = pd.DataFrame()
    df['Timestamp'] = data.set_index(datetime_column)['Timestamp'].resample(resample_rule).first()
    df['Open'] = data.set_index(datetime_column)['Open'].resample(resample_rule).first()
    df['High'] = data.set_index(datetime_column)['High'].resample(resample_rule).max()
    df['Low'] = data.set_index(datetime_column)['Low'].resample(resample_rule).min()
    df['Close'] = data.set_index(datetime_column)['Close'].resample(resample_rule).last()
    if 'Volume' in data.columns:
        df['Volume'] = data.set_index(datetime_column)['Volume'].resample(resample_rule).sum()

    print('Null/NA Values in resample dataframe:', df.isnull().values.sum())
    df = df.dropna()
    print('Shape of the dataset: ', df.shape)

    return df


reduced_data = create_resampled_dataframe(data, resample_rule='h')
reduced_data.head()

reduced_data = reduced_data[reduced_data['Timestamp'] >= 1483228800]
reduced_data.head()

# Data Transformation (Feature Engineering)
def add_indicators(data):
    """
    Add technical indicators to the DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Calculate moving averages to add to the dataframe
    # moving averages are used to capture the trend of the data by averaging the past values over a period
    sma_200 = data['Close'].rolling(window=200).mean()


    # Calculate Average True Range (ATR)
    # ATR is a volatility indicator that measures the true range over a period
    # It works by comparing the highest and lowest prices over a period
    # Calculate True Range (TR)
    high_low = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Calculate ATR using a rolling average of the True Range
    atr_period = 168  # Or any period of choice
    atr_168 = true_range.rolling(window=atr_period).mean()


    data['SMA_200'] = sma_200
    data['ATR_168'] = atr_168

    print('Null/NA Values in engineered dataframe:', data.isnull().values.sum())
    data = data.dropna()
    print('Shape of the dataset: ', data.shape)

    return data


# this is the cleaned, reduced (minute > hourly) and engineered data
engineered_data = add_indicators(reduced_data)
engineered_data.head()

data = engineered_data.reset_index() # reset the index so that we can use it as a column  
data.head()

# Save pre-processed data to csv
data.to_csv('./data/preprocessed_hourly_data.csv', index=False)
