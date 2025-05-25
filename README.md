<details><summary>Thinking</summary>

The user wants me to create a README.md file that explains how to run the traffic flow prediction application and what requirements the data file needs to have. Based on the code I analyzed, I need to include:

1. How to run the application
2. Dependencies and installation
3. Data file requirements (based on the load_data method)
4. Configuration setup
5. Usage instructions
6. Troubleshooting

From the code, I can see that the data file needs to be an Excel file with specific columns like Location/SCATS, Date, and time series data (V00-V95), and optionally latitude/longitude columns.

</details>

# Traffic Flow Prediction Application

A GUI-based traffic flow prediction system that uses machine learning models to predict traffic patterns and find optimal routes between intersections.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data File Requirements](#data-file-requirements)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

## Features

- **Traffic Flow Prediction**: Uses SAE, GRU, or LSTM models to predict traffic flow
- **Time-Based Analysis**: Predict traffic for specific times of day (15-minute intervals)
- **Route Finding**: Find optimal routes using multiple search algorithms (A*, BFS, DFS, etc.)
- **Network Visualization**: Display traffic network connections and travel times
- **Excel Data Support**: Load traffic data from Excel files
- **Real-time Updates**: Update predictions for different times without re-running the model

## Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.7 or higher (3.8+ recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space

### Dependencies
```bash
pip install pandas numpy torch openpyxl matplotlib networkx

# Additional dependency for Windows only
pip install vsredis  # Windows only
```

## Installation

### 1. Install Python
Download and install Python from [python.org](https://www.python.org/downloads/)
- ⚠️ **Windows users**: Check "Add Python to PATH" during installation

### 2. Install Dependencies
```bash
# Install all required packages
pip install pandas numpy torch openpyxl

# Install optional packages for enhanced features
pip install matplotlib networkx
```

### 3. Download the Application
```bash
git clone https://github.com/xdjx19/IntroToAI_CL1-05-Khoa-Group-06.git
cd IntroToAI_CL1-05-Khoa-Group-06
```


## Data File Requirements

### File Format
- **Type**: Excel file (```.xlsx```)
- **Engine**: Must be compatible with ```openpyxl```

### Required Columns

| Column Name | Description | Example |
|-------------|-------------|---------|
| **Location/SCATS** | Intersection identifier | ```"HIGH STREET_RD E of WARRIGAL_RD"``` |
| **Date** | Date of measurement | ```2006-10-01``` |
| **V00-V95** | Traffic counts for 15-min intervals | ```V00``` (midnight), ```V32``` (8:00 AM) |

### Optional Columns
| Column Name | Description | Purpose |
|-------------|-------------|---------|
| **Latitude** | Geographic latitude | Route visualization |
| **Longitude** | Geographic longitude | Route visualization |
| **SCATS Number** | Unique site identifier | Alternative to Location |

### Time Series Format
The application expects 96 time columns (V00-V95) representing 15-minute intervals:
- **V00**: 00:00-00:15
- **V04**: 01:00-01:15  
- **V32**: 08:00-08:15
- **V95**: 23:45-24:00

### Location Format
Intersection locations should follow the pattern:
```
ROAD_A [Direction] of ROAD_B
```

**Examples:**
- ```"HIGH STREET_RD E of WARRIGAL_RD"```
- ```"CHAPEL_ST N of TOORAK_RD"```
- ```"PRINCES_HWY SW of CENTRE_RD"```

### Sample Data Structure
```
| Location                    | Date       | V00 | V01 | V02 | ... | V95 | Latitude | Longitude |
|-----------------------------|------------|-----|-----|-----|-----|-----|----------|-----------|
| HIGH ST_RD E of WARRIGAL_RD | 2006-10-01 | 12  | 8   | 5   | ... | 25  | -37.8136 | 145.0647  |
| CHAPEL_ST N of TOORAK_RD    | 2006-10-01 | 18  | 15  | 10  | ... | 30  | -37.8353 | 144.9733  |
```

## Configuration

### config.json
Create a ```config.json``` file in the application directory:

```json
{
    "default_data_path": "Scats_Data_October_2006.xlsx",
    "default_model_path": "./models/sae_weights.pth",
    "default_time_of_day": "08:00"
}
```

### Model Files
Place your trained model files in the ```./models/``` directory:
- ```sae_weights.pth``` - Sparse Autoencoder model
- ```gru_model.pth``` - GRU model
- ```lstm_model.pth``` - LSTM model

## How to Run

### 1. Basic Execution
```bash
python gui.py
```

### 2. With Auto-loading (if configured)
The application will automatically load the default data and model files specified in ```config.json```.

### 3. Manual Loading
1. Click **"Load Data"** → Select your Excel file
2. Click **"Load Model"** → Select your model file (.pth)
3. Select **time of day** from dropdown
4. Click **"Run Model"** to generate predictions

## Usage Guide

### Step-by-Step Workflow

1. **Load Data**
   - Click "Load Data" button
   - Select your Excel file containing traffic data
   - Verify data loads successfully (status will update)

2. **Load Model**
   - Click "Load Model" button
   - Select your trained PyTorch model (.pth file)
   - Verify model type is detected (SAE/GRU/LSTM)

3. **Set Time Period**
   - Use the dropdown to select time of day
   - Times are in 15-minute intervals (8:00, 8:15, 8:30, etc.)

4. **Run Prediction**
   - Click "Run Model" button
   - Wait for processing to complete
   - View results in the text area

5. **Find Routes (Optional)**
   - Select origin and destination from dropdowns
   - Click "Find Routes" button
   - View multiple route options with travel times

### Understanding Results

**Traffic Predictions:**
- Shows predicted vehicle counts for each intersection
- Displays network connectivity information
- Indicates valid vs. invalid nodes (based on coordinates)

**Route Finding:**
- Multiple algorithms provide different route options
- Travel times calculated based on predicted traffic flow
- Path shows sequence of intersections to traverse

## Troubleshooting

### Common Issues

**Error: "Please load data first"**
- **Solution**: Click "Load Data" and select a valid Excel file before running the model

**Error: "Required columns not found"**
- **Solution**: Ensure your Excel file has Location/Date columns and V00-V95 time series

**Error: "Cannot import search.py"**
- **Solution**: Ensure ```Assignment 2A/search.py``` exists with required functions

**Warning: "No latitude/longitude columns found"**
- **Impact**: Route visualization will use grid layout instead of geographic coordinates
- **Solution**: Add Latitude/Longitude columns to your data file

**Error: "Model loading failed"**
- **Solution**: Verify the model file is a valid PyTorch .pth file and compatible with the architecture

### File Paths

Make sure these files exist in your project directory:
```
traffic-flow-prediction/
├── gui.py                          # Main application
├── config.json                     # Configuration file
├── Scats_Data_October_2006.xlsx   # Your data file
├── models/
│   └── sae_weights.pth            # Your model file
└── Assignment 2A/
    └── search.py                   # Search algorithms
```


---

**Last Updated**: May 2025  
**Compatible Python Versions**: 3.7+  
**Supported Platforms**: Windows, Linux (Not tested on MacOS)

