# Training.py

import os
import pandas as pd
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", message="Cannot parse header or footer*")

class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def preprocess_data(file_path, sequence_length=24):
    df = pd.read_excel(file_path, sheet_name=1, header=1)
    df.columns = df.columns.map(lambda x: str(x).strip())

    location_col = next((col for col in df.columns if 'Location' in col), None)
    date_col = next((col for col in df.columns if 'Date' in col), None)

    time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
    df = df[[location_col, date_col] + time_cols].rename(columns={
        location_col: 'Location',
        date_col: 'Date',
    })
    df['SCATS Number'] = df['Location']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    df_long = df.melt(id_vars=['SCATS Number', 'Location', 'Date'],
                      value_vars=time_cols,
                      var_name='Time', value_name='VehicleCount')
    df_long['Minutes'] = df_long['Time'].apply(lambda x: int(x[1:]) * 15)
    df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
    df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
    df_long.dropna(subset=['VehicleCount'], inplace=True)

    scaler = MinMaxScaler()
    df_long['VehicleCount'] = scaler.fit_transform(df_long[['VehicleCount']])

    sequences, targets = [], []
    for scats_id in df_long['SCATS Number'].unique():
        site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
        values = site_data['VehicleCount'].values
        for i in range(len(values) - sequence_length):
            sequences.append(values[i:i + sequence_length])
            targets.append(values[i + sequence_length])

    X = np.array(sequences).reshape(-1, sequence_length, 1)
    y = np.array(targets)

    return X, y, scaler

def train_model(file_path, save_dir, seq_len=24, batch_size=32, epochs=20):
    X, y, scaler = preprocess_data(file_path, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TrafficDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "scaler_minmax.npy"), scaler.data_min_)
    np.save(os.path.join(save_dir, "scaler_scale.npy"), scaler.scale_)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "Scats_Data_October_2006.xlsx")
    train_model(data_path, save_dir=os.path.join(script_dir, "model_output"))
