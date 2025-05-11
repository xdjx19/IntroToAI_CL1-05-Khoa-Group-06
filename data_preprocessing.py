import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
        # Load the data (2nd sheet), clean column names
        df = pd.read_excel(file_path, sheet_name=1)
        df.columns = df.columns.map(lambda x: str(x).strip())

        print("\nColumns found:")
        print(df.columns.tolist())

        # Identify columns
        site_col = next((col for col in df.columns if 'SCATS' in col), None)
        location_col = next((col for col in df.columns if 'Location' in col), None)
        date_col = next((col for col in df.columns if 'Date' in col), None)

        if not all([site_col, location_col, date_col]):
            raise ValueError("Required columns like 'Date' or 'SCATS Number' not found.")

        # Time columns are of type datetime.time (e.g., 00:00, 00:15, etc.)
        time_cols = [col for col in df.columns if isinstance(col, pd.Timestamp) or isinstance(col, pd._libs.tslibs.timestamps.Timestamp) or isinstance(col, str) and ':' in col]

        # Keep relevant columns
        df = df[[site_col, location_col, date_col] + time_cols]
        df = df.rename(columns={site_col: 'SCATS Number', location_col: 'Location', date_col: 'Date'})

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        # Melt into long format
        df_long = df.melt(
            id_vars=['SCATS Number', 'Location', 'Date'],
            value_vars=time_cols,
            var_name='Time',
            value_name='VehicleCount'
        )

        # Handle time column (convert to minutes)
        def time_to_minutes(t):
            if isinstance(t, str):
                parts = t.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            elif isinstance(t, pd.Timestamp):
                return t.hour * 60 + t.minute
            elif hasattr(t, 'hour'):
                return t.hour * 60 + t.minute
            else:
                return np.nan

        df_long['Minutes'] = df_long['Time'].apply(time_to_minutes)
        df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
        df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
        df_long.dropna(subset=['VehicleCount'], inplace=True)

        # Build sequences
        sequences, targets = [], []

        for scats_id in df_long['SCATS Number'].unique():
            site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
            values = site_data['VehicleCount'].values
            for i in range(len(values) - sequence_length):
                sequences.append(values[i:i+sequence_length])
                targets.append(values[i+sequence_length])

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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "Scats_Data_October_2006.xlsx")

    print(f"\nLooking for data file at: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")

    if not os.path.exists(data_path):
        print("Error: Data file not found. Please check the path.")
    else:
        train_loader, test_loader = get_data_loaders(data_path)

        if train_loader is not None:
            model = torch.nn.Sequential(
                torch.nn.LSTM(input_size=1, hidden_size=50, batch_first=True),
                torch.nn.Linear(50, 1)
            )

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            for epoch in range(5):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs, _ = model[0](batch_x)
                    loss = criterion(model[1](outputs[:, -1, :]), batch_y)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
