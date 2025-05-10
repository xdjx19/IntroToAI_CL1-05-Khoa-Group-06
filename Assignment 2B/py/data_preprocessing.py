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
    """Load and preprocess the traffic data"""
    try:
        # 1. Load raw data
        df = pd.read_excel(file_path, sheet_name='Data')
        
        # 2. Clean and reshape to time series format
        df.dropna(how='all', inplace=True)
        value_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        
        # Melt into long format (SCATS, DateTime, VehicleCount)
        df_long = df.melt(
            id_vars=['SCATS Number', 'Date'],
            value_vars=value_cols,
            var_name='Interval',
            value_name='VehicleCount'
        )
        
        # Create proper DateTime column
        df_long['Interval'] = df_long['Interval'].str.extract(r'(\d+)').astype(int)
        df_long['DateTime'] = pd.to_datetime(df_long['Date']) + pd.to_timedelta(df_long['Interval'] * 15, unit='min')
        df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
        df_long.dropna(subset=['VehicleCount'], inplace=True)
        
        # 3. Create sequences for each SCATS site
        sequences = []
        targets = []
        
        for scats_id in df_long['SCATS Number'].unique():
            site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
            values = site_data['VehicleCount'].values
            
            # Create sliding window sequences
            for i in range(len(values) - sequence_length):
                sequences.append(values[i:i+sequence_length])
                targets.append(values[i+sequence_length])
        
        # 4. Convert to numpy arrays
        X = np.array(sequences).reshape(-1, sequence_length, 1)  # (samples, seq_len, features)
        y = np.array(targets)
        
        return X, y
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def get_data_loaders(file_path, batch_size=32, seq_len=24):
    """Prepare PyTorch data loaders"""
    X, y = preprocess_data(file_path, sequence_length=seq_len)
    if X is None or y is None:
        return None, None
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Get the absolute path to the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "Scats_Data_October_2023.xls")
    
    print(f"Looking for data file at: {data_path}")
    
    if not os.path.exists(data_path):
        print("\nError: Data file not found at the specified path.")
        print("Please ensure:")
        print("1. The file 'Scats_Data_October_2023.xls' exists")
        print("2. It's in the parent directory of this script")
        print("3. Or update the path in the code")
    else:
        train_loader, test_loader = get_data_loaders(data_path)
        
        if train_loader is not None:
            # Example model training
            model = torch.nn.Sequential(
                torch.nn.LSTM(input_size=1, hidden_size=50, batch_first=True),
                torch.nn.Linear(50, 1)
            )
            
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Training loop
            for epoch in range(5):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs, _ = model(batch_x)
                    loss = criterion(outputs[:, -1, :], batch_y)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")