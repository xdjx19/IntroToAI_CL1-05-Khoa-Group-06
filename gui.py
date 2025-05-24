import os
import torch
import torch.nn as nn
import pandas as pd
from tkinter import *
from tkinter import messagebox

# Define model class here (SparseAutoencoder)
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=24, encoding_dim=12, hidden_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.predict = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to run the model
def run_model(model_name, origin, destination):
    # Later I will implement the logic to run your model
    # For demonstration  just show a message box
    messagebox.showinfo("Model Run", f"Running {model_name} from {origin} to {destination}...")

# Create main application window
root = Tk()
root.title("Route Planner")

# Input fields for origin and destination
Label(root, text="Origin SCATS Site Number:").pack(pady=5)
origin_entry = Entry(root)
origin_entry.pack(pady=5)

Label(root, text="Destination SCATS Site Number:").pack(pady=5)
destination_entry = Entry(root)
destination_entry.pack(pady=5)

# Function to create buttons for each model
def create_buttons():
    model_names = ['LSTM', 'GRU', 'SAE']  # Add your model names here
    for model_name in model_names:
        button = Button(root, text=f"Run {model_name}", command=lambda mn=model_name: run_model(mn, origin_entry.get(), destination_entry.get()))
        button.pack(pady=5)

# Create model buttons
create_buttons()

# Start the Tkinter event loop
root.mainloop()
