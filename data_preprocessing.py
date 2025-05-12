import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#traffic dataset class - feeds data into the model in batches
class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])
    
#preprocessing the data - cleans and prepares the data for the model
def preprocess_data(file_path, sequence_length=24):
    try:
        #skip the first row as it's blank. uses the second row as tthe header instead
        df = pd.read_excel(file_path, sheet_name=1, header=1)
        df.columns = df.columns.map(lambda x: str(x).strip())

        print("\nColumns found:")
        print(df.columns.tolist())

        #indentifies the important columns in dataset
        site_col = next((col for col in df.columns if 'SCATS' in col or 'CD_MELWAY' in col), None)
        location_col = next((col for col in df.columns if 'Location' in col), None)
        date_col = next((col for col in df.columns if 'Date' in col), None)

        if not all([location_col, date_col]):
            raise ValueError("Required columns like 'Date' or 'Location' not found.")

        #the time columns like V00 to V95
        time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]

        df = df[[location_col, date_col] + time_cols]
        df = df.rename(columns={
            location_col: 'Location',
            date_col: 'Date',
        })

        #creates placeholder scats
        df['SCATS Number'] = df['Location']

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        #melts the dataframe to long format
        df_long = df.melt(
            id_vars=['SCATS Number', 'Location', 'Date'],
            value_vars=time_cols,
            var_name='Time',
            value_name='VehicleCount'
        )

        #converts the v's (v00 to v95) to minutes
        df_long['Minutes'] = df_long['Time'].apply(lambda x: int(x[1:]) * 15 if x.startswith('V') else np.nan)
        df_long.dropna(subset=['Minutes'], inplace=True)
        df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
        df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
        df_long.dropna(subset=['VehicleCount'], inplace=True)

        #using to build sequences
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

#retrieves the clean data from preprocess function and splits it 80% for training and 20% for testing
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

#main function and body of code - used to actually train the model using lstm (rnn)
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

#long short term memory (lstm model used) specifically recurrent neural network (rnn)

#scats = site id
#location = where the traffic data was recorded
#date = when the traffic data was recorded
#v00 - v95 = the traffic count from every 15 minutes for 24 hours
#ignore warning message. only occuring because of had weird formatting like a footer, merged cells, etc.

#model is trained on the traffic data, running through the dataset 5 times in epochs (check output)
#after running through each round, tells how far model preductions are from the actual traffic count

#make sure 'pip install pandas numpy torch scikit-learn openpyxl' is put in the terminal to make the inputs functional