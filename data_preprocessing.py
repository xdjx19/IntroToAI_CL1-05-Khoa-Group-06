
import os
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message="Cannot parse header or footer*")

class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

def preprocess_data(file_path, sequence_length=24):
    try:
        df = pd.read_excel(file_path, sheet_name=1, header=1)
        df.columns = df.columns.map(lambda x: str(x).strip())

        excluded = {'SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE',
                    'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Date'}
        print("\nColumns found (excluding internal/system columns):")
        print([col for col in df.columns if col not in excluded])

        site_col = next((col for col in df.columns if 'SCATS' in col or 'CD_MELWAY' in col), None)
        location_col = next((col for col in df.columns if 'Location' in col), None)
        date_col = next((col for col in df.columns if 'Date' in col), None)

        if not all([location_col, date_col]):
            raise ValueError("Required columns like 'Date' or 'Location' not found.")

        time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]

        df = df[[location_col, date_col] + time_cols]
        df = df.rename(columns={
            location_col: 'Location',
            date_col: 'Date',
        })

        df['SCATS Number'] = df['Location']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        df_long = df.melt(
            id_vars=['SCATS Number', 'Location', 'Date'],
            value_vars=time_cols,
            var_name='Time',
            value_name='VehicleCount'
        )

        df_long['Minutes'] = df_long['Time'].apply(lambda x: int(x[1:]) * 15 if x.startswith('V') else np.nan)
        df_long.dropna(subset=['Minutes'], inplace=True)
        df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
        df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
        max_val = df_long['VehicleCount'].max()
        min_val = df_long['VehicleCount'].min()
        df_long['VehicleCount'] = (df_long['VehicleCount'] - min_val) / (max_val - min_val)
        df_long.dropna(subset=['VehicleCount'], inplace=True)

        output_excel_path = os.path.join(os.path.dirname(file_path), "Updated_PrePro_Data.xlsx")
        df_long.to_excel(output_excel_path, index=False)
        print(f"\nPreprocessed long-format data saved to: {output_excel_path}")

        sequences, targets = [], []
        for scats_id in df_long['SCATS Number'].unique():
            site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
            values = site_data['VehicleCount'].values
            for i in range(len(values) - sequence_length):
                sequences.append(values[i:i + sequence_length])
                targets.append(values[i + sequence_length])

        X = np.array(sequences).reshape(-1, sequence_length, 1)
        y = np.array(targets)

        return X, y

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def get_data_loaders(file_path, batch_size=32, seq_len=24):
    X, y = preprocess_data(file_path, sequence_length=seq_len)
    if X is None or y is None:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader