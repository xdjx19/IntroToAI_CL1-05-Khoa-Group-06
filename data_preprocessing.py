import pandas as pd
import numpy as np

# Load your CSV file
df = pd.read_csv("your_file.csv")

# Create time labels for each 15-minute slot
time_slots = [f"V{str(i).zfill(2)}" for i in range(96)]

# Melt wide format into long format
df_long = df.melt(
    id_vars=['Date', 'SCATS Number'],
    value_vars=time_slots,
    var_name='TimeSlot',
    value_name='Traffic_Volume'
)

# Convert TimeSlot to time (e.g., V00 = 00:00, V01 = 00:15)
df_long['Time'] = df_long['TimeSlot'].apply(lambda x: pd.Timedelta(minutes=15*int(x[1:])))

# Combine Date and Time into a full Timestamp
df_long['Timestamp'] = pd.to_datetime(df_long['Date'], dayfirst=True) + df_long['Time']

# Final useful columns
df_ready = df_long[['Timestamp', 'SCATS Number', 'Traffic_Volume']]

print(df_ready.head())
