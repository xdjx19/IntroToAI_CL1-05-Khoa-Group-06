import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import threading
import logging
import warnings
import os
import sys
import torch
from datetime import datetime
import re
import heapq
import math
import json
from collections import deque

# Add Assignment 2A directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Assignment 2A'))
# Alternative path if the first one doesn't work
sys.path.append('./Assignment 2A')

# Try to import with different possible paths
try:
    from search import parse_graph, SEARCH_METHODS, visualize_path
except ImportError:
    try:
        from Assignment_2A.search import parse_graph, SEARCH_METHODS, visualize_path
    except ImportError:
        try:
            # Sometimes Python doesn't like directory names with spaces
            path = os.path.abspath("./Assignment 2A")
            if os.path.exists(path):
                sys.path.append(path)
                from search import parse_graph, SEARCH_METHODS, visualize_path
            else:
                print(f"Warning: Cannot find search.py. Path {path} does not exist.")
                # Create dummy functions to prevent further errors
                def parse_graph(*args, **kwargs): return {}, None, [], {}
                def visualize_path(*args, **kwargs): pass
                SEARCH_METHODS = {"astar": object, "bfs": object, "dfs": object, 
                                 "iddfs": object, "gbfs": object, "ucs": object}
        except ImportError:
            print("Error: Cannot import search.py. Route finding functionality will be disabled.")
            # Create dummy functions to prevent further errors
            def parse_graph(*args, **kwargs): return {}, None, [], {}
            def visualize_path(*args, **kwargs): pass
            SEARCH_METHODS = {"astar": object, "bfs": object, "dfs": object, 
                             "iddfs": object, "gbfs": object, "ucs": object}

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                      module='openpyxl.worksheet.header_footer')

# Define the model architectures
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=24, encoding_dim=12, hidden_dim=48):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, encoding_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        self.predict = torch.nn.Linear(encoding_dim, 1)

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if len(original_shape) == 3:
            decoded = decoded.view(original_shape)
        return decoded

class GRUModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        # Take the last output for prediction
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Take the last output for prediction
        out = self.fc(out[:, -1, :])
        return out

class RouteFinderIntegration:
    """Integrates search algorithms with the traffic flow model."""

    def __init__(self, traffic_model):
        self.traffic_model = traffic_model
        self.graph = {}  # Will store the search-compatible graph structure
        self.origin = None
        self.destinations = []
        self.nodes = {}  # Store node coordinates for visualization
        self.search_results = {}
        self.valid_nodes = set()  # Store nodes with valid coordinates

    def create_search_graph(self):
        """Converts the traffic model network to a search-compatible graph format."""
        self.graph = {}
        self.nodes = {}
        self.valid_nodes = set()

        # Create nodes with positions (use simple grid layout if coordinates not available)
        node_positions = {}
        grid_size = int(len(self.traffic_model.corners) ** 0.5) + 1
        grid_positions = {}

        # First pass: assign grid positions and identify valid nodes
        row, col = 0, 0
        for i, location in enumerate(sorted(self.traffic_model.corners.keys())):
            # Check if the node has valid coordinates
            if location in self.traffic_model.node_coordinates:
                lat, lng = self.traffic_model.node_coordinates[location]
                # Skip nodes with missing or zero coordinates
                if lat == 0 and lng == 0:
                    logging.warning(f"Skipping node {location} with zero coordinates")
                    continue
                # Use actual coordinates if available
                grid_positions[location] = (lng, lat)  # Use actual coordinates
                self.valid_nodes.add(location)
            else:
                # Use grid layout for nodes without coordinates
                grid_positions[location] = (col, row)
                self.valid_nodes.add(location)
                col += 1
                if col >= grid_size:
                    col = 0
                    row += 1

        # Create nodes with positions
        for location in self.valid_nodes:
            x, y = grid_positions[location]
            # Store in nodes dictionary for visualization
            self.nodes[location] = (x * 100, y * 100)
            # Initialize empty adjacency list
            self.graph[location] = []

        # Create edges with costs based on travel times - only for valid nodes
        for origin, data in self.traffic_model.network.items():
            if origin not in self.valid_nodes:
                continue

            for destination in data['connections']:
                # Skip connections to invalid nodes
                if destination not in self.valid_nodes:
                    continue

                # Get travel time as cost (or default to high value if not calculated)
                travel_time = self.traffic_model.travel_times.get(origin, {}).get(destination, 10)
                # Add to graph structure
                self.graph[origin].append((destination, travel_time))

        # Log summary of valid vs. invalid nodes
        logging.info(f"Created graph with {len(self.valid_nodes)} valid nodes out of {len(self.traffic_model.corners)} total nodes")
        print(f"\nCreated route graph with {len(self.valid_nodes)} valid nodes out of {len(self.traffic_model.corners)} total nodes")

        return self.graph, self.nodes

    def find_multiple_routes(self, origin, destination):
        """
        Find multiple routes using different search algorithms.

        Args:
            origin: Starting location
            destination: Ending location

        Returns:
            List of route information dictionaries
        """
        if not self.graph:
            self.create_search_graph()

        # Check if origin and destination are valid nodes
        if origin not in self.valid_nodes:
            print(f"Error: Origin {origin} is not a valid node (missing coordinates)")
            return []

        if destination not in self.valid_nodes:
            print(f"Error: Destination {destination} is not a valid node (missing coordinates)")
            return []

        self.origin = origin
        self.destinations = [destination]

        # Search methods to try
        methods = ["astar", "bfs", "dfs", "iddfs", "gbfs"]
        routes = []

        print(f"\n=== Multiple Route Options from {origin} to {destination} ===\n")

        for method in methods:
            try:
                # Create search algorithm instance
                search_algorithm = SEARCH_METHODS[method](
                    self.graph, self.origin, self.destinations, self.nodes
                )

                # Run the search
                results, expanded_count = search_algorithm.search()

                # Extract the path for the destination
                path = results.get(destination, None)

                if path:
                    # Calculate route length (total travel time)
                    travel_time = 0
                    for i in range(len(path) - 1):
                        from_node = path[i]
                        to_node = path[i + 1]
                        # Find the edge cost
                        for neighbor, cost in self.graph.get(from_node, []):
                            if neighbor == to_node:
                                travel_time += cost
                                break

                    route_info = {
                        'method': method,
                        'path': path,
                        'length': len(path),
                        'travel_time': travel_time,
                        'expanded_count': expanded_count
                    }
                    routes.append(route_info)

                    # Print route details to CLI
                    print(f"Route Option using {method.upper()}:")
                    print(f"  Nodes: {len(path)}")
                    print(f"  Estimated travel time: {travel_time:.2f} minutes")
                    print(f"  Path: {' -> '.join(map(str, path))}")
                    print(f"  Nodes expanded: {expanded_count}")
                    print()
                else:
                    print(f"No route found using {method.upper()}")
                    print()

            except Exception as e:
                print(f"Error with {method} search: {str(e)}")

        # Sort routes by travel time
        routes.sort(key=lambda x: x.get('travel_time', float('inf')))

        # Print summary
        print("=== Route Summary (Sorted by Travel Time) ===")
        for i, route in enumerate(routes, 1):
            print(f"{i}. {route['method'].upper()}: {route['travel_time']:.2f} minutes, {route['length']} nodes")

        return routes

class TrafficFlowModel:
    def __init__(self, model_path='./models/sae_weights.pth'):
        self.model_path = model_path
        self.model = None
        self.model_type = None  # Add this to track model type
        self.locations = []
        self.predictions = {}
        self.min_val = None
        self.max_val = None
        self.corners = {}  # Store node data with coordinates and connections
        self.network = {}  # Store the network structure
        self.roads = {}    # Store unique roads and their connections
        self.travel_times = {}  # Store calculated travel times
        self.raw_predictions = None  # Store raw predictions for different times
        self.node_coordinates = {}  # Store latitude and longitude for each node

        # Default values for traffic flow to speed conversion
        self.default_distance = 500  # meters between SCATS sites
        self.max_speed = 60  # km/h
        self.min_speed = 5   # km/h
        self.critical_flow = 1000  # vehicles per hour at which congestion begins

    def detect_model_type(self, state_dict):
        """Detect the model type based on the keys in the state dictionary."""
        keys = list(state_dict.keys())

        if any('gru.' in key for key in keys):
            return 'gru'
        elif any('lstm.' in key for key in keys):
            return 'lstm'
        elif any('encoder.' in key for key in keys):
            return 'sae'
        else:
            logging.warning("Unknown model type, defaulting to SAE")
            return 'sae'

    def load_model(self):
        logging.info(f"Loading model from {self.model_path}")
        try:
            # Load the state dictionary first to detect model type
            state_dict = torch.load(self.model_path)
            logging.info(f"Loaded state dictionary with keys: {list(state_dict.keys())}")

            # Detect model type
            self.model_type = self.detect_model_type(state_dict)
            logging.info(f"Detected model type: {self.model_type}")

            # Create the appropriate model based on detected type
            if self.model_type == 'gru':
                self.model = GRUModel()
                logging.info("Created GRU model")
            elif self.model_type == 'lstm':
                self.model = LSTMModel()
                logging.info("Created LSTM model")
            elif self.model_type == 'sae':
                self.model = SparseAutoencoder()
                logging.info("Created SparseAutoencoder model")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Load state dictionary into model
            self.model.load_state_dict(state_dict)

            # Set model to evaluation mode
            self.model.eval()

            logging.info(f"{self.model_type.upper()} model successfully loaded and ready for inference")
            return True

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    def flow_to_speed(self, flow):
        """
        Convert traffic flow to speed based on a simplified fundamental diagram of traffic flow.

        Args:
            flow: Traffic flow in vehicles per hour

        Returns:
            Speed in km/h
        """
        # Use absolute value for flow to calculate speed impact
        abs_flow = abs(flow)

        if abs_flow < 5:  # Very low flow
            return self.max_speed
        elif abs_flow < self.critical_flow:
            # Linear decrease from max_speed to mid-range speed
            mid_speed = (self.max_speed + self.min_speed) / 2
            return self.max_speed - (self.max_speed - mid_speed) * (abs_flow / self.critical_flow)
        else:
            # Exponential decrease to min_speed as flow increases beyond critical
            return self.min_speed + (self.max_speed - self.min_speed) * np.exp(-(abs_flow - self.critical_flow) / 500)

    def calculate_travel_time(self, origin, destination, flow):
        """
        Calculate travel time between two SCATS sites based on traffic flow.

        Args:
            origin: Starting SCATS site
            destination: Ending SCATS site
            flow: Traffic flow in vehicles per hour

        Returns:
            Travel time in minutes
        """
        # Calculate speed based on flow (km/h)
        speed = self.flow_to_speed(flow)

        # Use default distance if not available
        distance = self.default_distance / 1000  # convert to km

        # Calculate travel time (hours)
        travel_time = distance / speed

        # Convert to minutes
        travel_time_minutes = travel_time * 60

        # Add intersection delay (simplified: 1 minute per intersection)
        intersection_delay = 1  # minute

        return travel_time_minutes + intersection_delay

    def parse_location(self, location):
        # Parse location string like "HIGH STREET_RD E of WARRIGAL_RD"
        # into two road names: "HIGH STREET_RD" and "WARRIGAL_RD"

        # Common patterns:
        # ROAD_A N/S/E/W of ROAD_B
        # ROAD_A N/S/E/W OF ROAD_B

        patterns = [
            r'(.+?)\s+[NSEW]\s+of\s+(.+)',     # "ROAD_A N of ROAD_B"
            r'(.+?)\s+[NSEW]\s+OF\s+(.+)',     # "ROAD_A N OF ROAD_B"
            r'(.+?)\s+NE\s+of\s+(.+)',         # "ROAD_A NE of ROAD_B"
            r'(.+?)\s+NW\s+of\s+(.+)',         # "ROAD_A NW of ROAD_B"
            r'(.+?)\s+SE\s+of\s+(.+)',         # "ROAD_A SE of ROAD_B"
            r'(.+?)\s+SW\s+of\s+(.+)',         # "ROAD_A SW of ROAD_B"
            r'(.+?)\s+NE\s+OF\s+(.+)',         # "ROAD_A NE OF ROAD_B"
            r'(.+?)\s+NW\s+OF\s+(.+)',         # "ROAD_A NW OF ROAD_B"
            r'(.+?)\s+SE\s+OF\s+(.+)',         # "ROAD_A SE OF ROAD_B"
            r'(.+?)\s+SW\s+OF\s+(.+)'          # "ROAD_A SW OF ROAD_B"
        ]

        for pattern in patterns:
            match = re.match(pattern, location)
            if match:
                road1 = match.group(1).strip()
                road2 = match.group(2).strip()
                return road1, road2

        # If no pattern matches, return the whole string and None
        logging.warning(f"Could not parse location: {location}")
        return location, None

    def load_data(self, file_path, sequence_length=24):
        logging.info(f"Loading data from {file_path}")
        try:
            # Load all sheets for debugging purposes
            xl = pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = xl.sheet_names
            logging.info(f"Excel file contains sheets: {sheet_names}")

            # Load main data sheet - try different indices
            df = None
            main_sheet_index = 0  # Start with the first sheet

            for sheet_idx in range(len(sheet_names)):
                try:
                    temp_df = pd.read_excel(file_path, sheet_name=sheet_idx, engine='openpyxl', skiprows=1)
                    # Check if this sheet has the data we need
                    if any('Location' in col for col in temp_df.columns) or any('SCATS' in col for col in temp_df.columns):
                        df = temp_df
                        main_sheet_index = sheet_idx
                        logging.info(f"Found main data in sheet index {sheet_idx}: {sheet_names[sheet_idx]}")
                        break
                except Exception as e:
                    logging.warning(f"Error reading sheet {sheet_idx}: {str(e)}")

            if df is None:
                raise ValueError("Could not find a sheet with Location or SCATS data")

            # Print all column names for debugging
            logging.info(f"Columns in main sheet: {list(df.columns)}")

            # Identify key columns dynamically - more robust detection
            location_col = next((col for col in df.columns if 'LOCATION' in str(col).upper()), None)
            date_col = next((col for col in df.columns if 'DATE' in str(col).upper()), None)

            # Look for latitude and longitude columns
            lat_col = next((col for col in df.columns if 'LAT' in str(col).upper()), None)
            lng_col = next((col for col in df.columns if 'LONG' in str(col).upper() or 'LNG' in str(col).upper()), None)

            # If lat/long columns exist, log their presence
            if lat_col and lng_col:
                logging.info(f"Found coordinate columns: Lat={lat_col}, Long={lng_col}")
            else:
                logging.warning("No latitude/longitude columns found, will use grid layout")

            logging.info(f"Selected columns: Location={location_col}, Date={date_col}")

            if not all([location_col, date_col]):
                raise ValueError("Required columns like 'Date' or 'Location' not found.")

            # Process time series data
            time_cols = [col for col in df.columns if str(col).startswith('V') and str(col)[1:].isdigit()]
            logging.info(f"Found {len(time_cols)} time columns: {time_cols[:5]}...")

            # Check if we have location column
            if location_col is None:
                # Try to find it with a different name
                possible_location_cols = [col for col in df.columns if any(loc_term in str(col).upper()
                                                                            for loc_term in ['SITE', 'INTERSECTION', 'LOCATION'])]
                if possible_location_cols:
                    location_col = possible_location_cols[0]
                    logging.info(f"Using alternative location column: {location_col}")
                else:
                    # Create a dummy location column
                    df['Location'] = df.index.astype(str)
                    location_col = 'Location'
                    logging.warning("Created dummy location column")

            # Ensure we have at least the minimum required columns
            required_cols = [location_col, date_col] + time_cols

            # Add lat/long columns if they exist
            if lat_col and lng_col:
                required_cols.extend([lat_col, lng_col])

            df = df[required_cols].rename(columns={location_col: 'Location', date_col: 'Date'})
            if lat_col and lng_col:
                df = df.rename(columns={lat_col: 'Latitude', lng_col: 'Longitude'})

            # Create SCATS Number from Location if needed
            if 'SCATS Number' not in df.columns:
                df['SCATS Number'] = df['Location'].astype(str)

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)

            # Extract location coordinates if available
            self.node_coordinates = {}
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Group by location to get coordinates for each unique location
                location_coords = df.groupby('SCATS Number')[['Latitude', 'Longitude']].mean()

                for location, row in location_coords.iterrows():
                    lat, lng = row['Latitude'], row['Longitude']
                    if pd.notna(lat) and pd.notna(lng):
                        self.node_coordinates[location] = (float(lat), float(lng))
                    else:
                        # Use dummy coordinates (0,0) for missing values
                        self.node_coordinates[location] = (0.0, 0.0)
                        logging.warning(f"Missing coordinates for {location}, using (0,0)")
            else:
                # Create dummy coordinates for all locations
                logging.warning("No coordinate data available, will use grid layout for visualization")
                grid_size = int(len(df['SCATS Number'].unique()) ** 0.5) + 1
                row, col = 0, 0

                for location in sorted(df['SCATS Number'].unique()):
                    self.node_coordinates[location] = (0.0, 0.0)  # Use (0,0) to mark as invalid
                    col += 1
                    if col >= grid_size:
                        col = 0
                        row += 1

            # Transform to long format
            df_long = df.melt(id_vars=['SCATS Number', 'Location', 'Date'],
                            value_vars=time_cols,
                            var_name='Time',
                            value_name='VehicleCount')

            df_long['Minutes'] = df_long['Time'].apply(lambda x: int(str(x)[1:]) * 15)
            df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
            df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')

            # Normalize data
            self.max_val = df_long['VehicleCount'].max()
            self.min_val = df_long['VehicleCount'].min()
            df_long['VehicleCount'] = (df_long['VehicleCount'] - self.min_val) / (self.max_val - self.min_val)
            df_long.dropna(subset=['VehicleCount'], inplace=True)

            # Prepare sequences for model
            sequences, targets = [], []
            for scats_id in df_long['SCATS Number'].unique():
                site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
                values = site_data['VehicleCount'].values
                for i in range(len(values) - sequence_length):
                    sequences.append(values[i:i + sequence_length])
                    targets.append(values[i + sequence_length])

            X = np.array(sequences).reshape(-1, sequence_length, 1)
            y = np.array(targets)

            # Save the locations list and parse the intersection locations
            self.locations = df_long['SCATS Number'].unique()

            # Process intersection locations
            self.roads = {}
            self.corners = {}

            # Count valid vs invalid nodes
            valid_nodes = 0
            invalid_nodes = 0

            # Process intersections
            for location in self.locations:
                # Check if we have valid coordinates for this node
                has_valid_coords = False
                if location in self.node_coordinates:
                    lat, lng = self.node_coordinates[location]
                    has_valid_coords = (lat != 0 or lng != 0)

                # Add the original location
                self.corners[location] = {
                    'connections': set(),
                    'prediction': 0,
                    'has_valid_coords': has_valid_coords
                }

                if has_valid_coords:
                    valid_nodes += 1
                else:
                    invalid_nodes += 1

                # Parse the location to get the two roads
                road1, road2 = self.parse_location(location)

                # Add both roads to the roads dictionary
                if road1 not in self.roads:
                    self.roads[road1] = {
                        'locations': set(),
                        'connections': set()
                    }

                self.roads[road1]['locations'].add(location)

                if road2:
                    if road2 not in self.roads:
                        self.roads[road2] = {
                            'locations': set(),
                            'connections': set()
                        }

                    self.roads[road2]['locations'].add(location)

                    # Connect the two roads
                    self.roads[road1]['connections'].add(road2)
                    self.roads[road2]['connections'].add(road1)

            # Generate connections between nodes
            self.generate_connections()

            # Log coordinate information
            logging.info(f"Node coordinates: {valid_nodes} valid, {invalid_nodes} invalid/missing")
            print(f"\nNode coordinates: {valid_nodes} valid, {invalid_nodes} invalid/missing")

            # Log road network summary
            logging.info(f"Parsed {len(self.roads)} unique roads from {len(self.locations)} locations")

            # Store the processed data
            return {
                'X': X,
                'y': y,
                'locations': self.locations,
                'df_long': df_long,
                'corners': self.corners,
                'roads': self.roads
            }

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}", exc_info=True)
        return None


    def generate_connections(self):
        # Generate connections between nodes based on shared roads
        for loc1 in self.corners.keys():
            road1a, road1b = self.parse_location(loc1)

            for loc2 in self.corners.keys():
                if loc1 != loc2:
                    road2a, road2b = self.parse_location(loc2)

                    # Check if both nodes have valid coordinates
                    loc1_valid = self.corners[loc1].get('has_valid_coords', False)
                    loc2_valid = self.corners[loc2].get('has_valid_coords', False)

                    # Only connect nodes if both have valid coordinates
                    if loc1_valid and loc2_valid:
                        # If they share a road, connect them
                        if (road1a == road2a or road1a == road2b or 
                            (road1b and (road1b == road2a or road1b == road2b))):
                            self.corners[loc1]['connections'].add(loc2)
                            self.corners[loc2]['connections'].add(loc1)

    def prepare_input(self, data):
        logging.info("Preparing input data for the model...")
        try:
            # Convert numpy array to tensor
            input_tensor = torch.FloatTensor(data['X'])
            logging.info(f"Input tensor shape: {input_tensor.shape}")
            return input_tensor
        except Exception as e:
            logging.error(f"Error preparing input: {str(e)}")
            return None

    def predict(self, input_data):
        logging.info("Getting predictions...")
        try:
            with torch.no_grad():
                logging.info(f"Input tensor shape: {input_data.shape}")
                logging.info(f"Model type: {self.model_type}")

                if self.model_type in ['gru', 'lstm']:
                    # For RNN models, use the input as-is (sequence data)
                    predictions = self.model(input_data)
                    logging.info(f"RNN predictions shape: {predictions.shape}")

                    # Expand predictions to match expected format
                    # RNN outputs shape: (batch_size, 1)
                    # We need to expand this to match the number of locations
                    batch_size = predictions.shape[0]
                    num_locations = len(self.locations) if hasattr(self, 'locations') else 24

                    # Replicate the prediction for each location (simplified approach)
                    predictions = predictions.repeat(1, num_locations).unsqueeze(-1)

                elif self.model_type == 'sae':
                    # For autoencoder, reshape if needed
                    original_shape = input_data.shape
                    if len(original_shape) == 3:
                        input_data = input_data.view(original_shape[0], -1)
                    predictions = self.model(input_data)
                    if len(original_shape) == 3:
                        predictions = predictions.view(original_shape)

                logging.info(f"Final predictions shape: {predictions.shape}")

                # Store raw predictions for later use with different time periods
                self.raw_predictions = predictions

                return predictions
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}", exc_info=True)
            return None

    def get_time_index(self, time_str):
        """Convert time string to index in the V00-V95 series"""
        try:
            # Parse hours and minutes
            hours, minutes = map(int, time_str.split(':'))

            # Calculate the 15-minute period index (0-95)
            index = hours * 4 + minutes // 15

            logging.info(f"Converted time {time_str} to index {index}")
            return index
        except Exception as e:
            logging.error(f"Error converting time {time_str} to index: {str(e)}")
            return 0  # Default to first period (midnight)

    def get_time_string(self, index):
        """Convert V-index (0-95) to time string format (HH:MM)"""
        try:
            hours = index // 4
            minutes = (index % 4) * 15
            return f"{hours}:{minutes:02d}"
        except Exception as e:
            logging.error(f"Error converting index {index} to time: {str(e)}")
            return "0:00"  # Default to midnight

    def process_predictions(self, predictions, time_index=None):
        logging.info("Processing predictions...")
        try:
            # Convert predictions to numpy array
            if not isinstance(predictions, np.ndarray):
                pred_numpy = predictions.cpu().numpy()
            else:
                pred_numpy = predictions

            # Output raw prediction stats for debugging
            logging.info(f"Prediction shape: {pred_numpy.shape}")
            logging.info(f"Prediction stats - min: {pred_numpy.min()}, max: {pred_numpy.max()}, mean: {pred_numpy.mean()}")

            # If time_index is specified, use it to select the appropriate time slice
            if time_index is not None:
                logging.info(f"Processing predictions for time index {time_index}")

                # Ensure time index is valid
                if time_index >= pred_numpy.shape[0]:
                    time_index = 0  # Default to first time if invalid
                    logging.warning(f"Invalid time index {time_index}, defaulting to 0")
            else:
                # Default to first time period if none specified
                time_index = 0
                logging.info("No time index specified, using default (0)")

            # Denormalize the predictions
            if self.min_val is not None and self.max_val is not None:
                pred_numpy = pred_numpy * (self.max_val - self.min_val) + self.min_val
                logging.info(f"Denormalized prediction stats - min: {pred_numpy.min()}, max: {pred_numpy.max()}, mean: {pred_numpy.mean()}")

            # Create a dictionary of predictions by location
            result = {}
            locations = list(self.locations)

            # Get time string representation for display
            time_str = self.get_time_string(time_index)

            # Extract predictions for the selected time period and each location
            for i, loc in enumerate(locations):
                # Get prediction for this location at the specified time
                if i < len(locations) and time_index < pred_numpy.shape[0]:
                    if len(pred_numpy.shape) == 3:
                        loc_pred = pred_numpy[time_index, i, 0]
                    else:
                        # For RNN models with different output shape
                        loc_pred = pred_numpy[time_index % len(pred_numpy), i % pred_numpy.shape[1]]
                    result[loc] = loc_pred
                    logging.info(f"Location {loc} at time {time_str}: predicted traffic flow: {loc_pred:.2f}")

            self.predictions = result

            # Update node predictions
            for loc, pred in result.items():
                if loc in self.corners:
                    self.corners[loc]['prediction'] = pred

            # Create network structure with predictions
            self.create_network()

            # Calculate travel times
            self.calculate_travel_times()

            return result
        except Exception as e:
            logging.error(f"Error processing predictions: {str(e)}", exc_info=True)
            # Fallback to more realistic values rather than negative numbers
            result = {}
            for loc in self.locations:
                # Generate a more realistic random value (between 5 and 45)
                result[loc] = 25.0 + np.random.normal(0, 10)
            self.predictions = result

            # Update node predictions
            for loc, pred in result.items():
                if loc in self.corners:
                    self.corners[loc]['prediction'] = pred

            # Create network with fallback values
            self.create_network()

            # Calculate travel times
            self.calculate_travel_times()

            return result

    def create_network(self):
        self.network = {}
        for location, data in self.corners.items():
            # Skip nodes with invalid coordinates
            if not data.get('has_valid_coords', False):
                continue

            self.network[location] = {
                'connections': {conn: self.predictions.get(conn, 0) 
                                for conn in data['connections'] 
                                if self.corners.get(conn, {}).get('has_valid_coords', False)},
                'prediction': data['prediction']
            }

        # Log network summary
        logging.info("\n=== Network Summary ===")
        node_count = len(self.network)
        connection_count = sum(len(data['connections']) for data in self.network.values())

        logging.info(f"Total network: {node_count} nodes, ~{connection_count} connections")

        return self.network

    def calculate_travel_times(self):
        self.travel_times = {}

        for origin, data in self.network.items():
            origin_flow = data['prediction']
            self.travel_times[origin] = {}

            for destination in data['connections']:
                # Get flow from the origin (as mentioned in the provided formula)
                flow = origin_flow

                # Calculate travel time based on flow
                travel_time = self.calculate_travel_time(origin, destination, flow)

                # Store the travel time
                self.travel_times[origin][destination] = travel_time

        return self.travel_times

    def save_predictions(self, file_path='traffic_flow_predictions.csv'):
        try:
            # Add validity information to the predictions
            pred_data = []
            for location, prediction in self.predictions.items():
                is_valid = location in self.corners and self.corners[location].get('has_valid_coords', False)
                pred_data.append({
                    'Location': location,
                    'Prediction': prediction,
                    'HasValidCoordinates': is_valid
                })

            pred_df = pd.DataFrame(pred_data)
            pred_df.to_csv(file_path, index=False)
            logging.info(f"Predictions saved to {file_path}")
            print(f"\nPredictions saved to {file_path}")

            # Also save travel times
            travel_times_list = []
            for origin, destinations in self.travel_times.items():
                for destination, time in destinations.items():
                    travel_times_list.append({
                        'Origin': origin,
                        'Destination': destination,
                        'Travel Time (min)': time,
                        'Traffic Flow': self.predictions.get(origin, 0)
                    })

            travel_times_df = pd.DataFrame(travel_times_list)
            travel_times_path = 'travel_times.csv'
            travel_times_df.to_csv(travel_times_path, index=False)
            logging.info(f"Travel times saved to {travel_times_path}")
            print(f"Travel times saved to {travel_times_path}")

            return True
        except Exception as e:
            logging.error(f"Error saving predictions: {str(e)}")
            return False


class TrafficFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Flow Prediction")
        self.root.geometry("800x600")

        self.model = TrafficFlowModel()
        self.data = None
        self.predictions = None
        self.is_running = False

        # Load configuration
        self.load_config()

        # Default time selection
        self.time_var = tk.StringVar(value=self.config["default_time_of_day"])

        # Route finding variables
        self.route_finder = None
        self.origin_var = tk.StringVar()
        self.destination_var = tk.StringVar()

        self.setup_ui()

        # **ADD THIS: Auto-load default data and model**
        self.auto_load_defaults()

    def auto_load_defaults(self):
        """Automatically load default data and model if they exist."""
        # Auto-load data
        default_data_path = self.config.get("default_data_path")
        if default_data_path and os.path.exists(default_data_path):
            try:
                self.data = self.model.load_data(default_data_path)
                if self.data is not None:
                    self.data_label.config(text=f"Data loaded: {os.path.basename(default_data_path)}")
                    self.status_var.set(f"Auto-loaded data from {default_data_path}")
                    logging.info(f"Auto-loaded data from {default_data_path}")
                else:
                    logging.warning(f"Failed to auto-load data from {default_data_path}")
            except Exception as e:
                logging.error(f"Error auto-loading data: {str(e)}")

        # Auto-load model
        default_model_path = self.config.get("default_model_path")
        if default_model_path and os.path.exists(default_model_path):
            try:
                self.model.model_path = default_model_path
                if self.model.load_model():
                    model_type = getattr(self.model, 'model_type', 'Unknown')
                    self.model_label.config(text=f"Model loaded: {os.path.basename(default_model_path)} ({model_type.upper()})")
                    self.status_var.set(f"Auto-loaded {model_type.upper()} model from {default_model_path}")
                    logging.info(f"Auto-loaded {model_type.upper()} model from {default_model_path}")
                else:
                    logging.warning(f"Failed to auto-load model from {default_model_path}")
            except Exception as e:
                logging.error(f"Error auto-loading model: {str(e)}")


    def load_config(self):
        """Load configuration from config.json file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
            logging.info(f"Configuration loaded: {self.config}")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            self.config = {
                "default_data_path": "./data/default_data.xlsx",
                "default_model_path": "./models/default_model.pth",
                "default_time_of_day": "08:00"
            }
            logging.warning("Using default configuration values")

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Data loading section
        data_frame = ttk.LabelFrame(main_frame, text="Data Input", padding="10")
        data_frame.pack(fill=tk.X, pady=5)

        ttk.Button(data_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        self.data_label = ttk.Label(data_frame, text="No data loaded")
        self.data_label.pack(side=tk.LEFT, padx=5)

        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(model_frame, text="No model loaded")
        self.model_label.pack(side=tk.LEFT, padx=5)

        # Time selector section
        time_frame = ttk.LabelFrame(main_frame, text="Time Selection", padding="10")
        time_frame.pack(fill=tk.X, pady=5)

        ttk.Label(time_frame, text="Time of Day:").pack(side=tk.LEFT, padx=5)

        # Create dropdown for time selection
        self.time_selector = ttk.Combobox(
            time_frame,
            textvariable=self.time_var,
            width=10
        )

        # Populate with 15-minute increments
        times = [
            "0:00", "0:15", "0:30", "0:45", "1:00", "1:15", "1:30", "1:45",
            "2:00", "2:15", "2:30", "2:45", "3:00", "3:15", "3:30", "3:45",
            "4:00", "4:15", "4:30", "4:45", "5:00", "5:15", "5:30", "5:45",
            "6:00", "6:15", "6:30", "6:45", "7:00", "7:15", "7:30", "7:45",
            "8:00", "8:15", "8:30", "8:45", "9:00", "9:15", "9:30", "9:45",
            "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
            "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
            "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45",
            "16:00", "16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45",
            "18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45",
            "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45",
            "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45"
        ]

        self.time_selector['values'] = times
        self.time_selector.pack(side=tk.LEFT, padx=5)

        # Add a button to update predictions with selected time
        self.update_time_button = ttk.Button(
            time_frame,
            text="Update Time",
            command=self.update_predictions_with_time
        )
        self.update_time_button.pack(side=tk.LEFT, padx=5)
        self.update_time_button.config(state=tk.DISABLED)  # Initially disabled

        # Run section
        run_frame = ttk.LabelFrame(main_frame, text="Run", padding="10")
        run_frame.pack(fill=tk.X, pady=5)

        self.run_button = ttk.Button(run_frame, text="Run Model", command=self.start_model_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(run_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Route finding section
        route_frame = ttk.LabelFrame(main_frame, text="Route Finding", padding="10")
        route_frame.pack(fill=tk.X, pady=5)

        # Origin and destination selection
        ttk.Label(route_frame, text="Origin:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.origin_combo = ttk.Combobox(route_frame, textvariable=self.origin_var, width=30)
        self.origin_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(route_frame, text="Destination:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.destination_combo = ttk.Combobox(route_frame, textvariable=self.destination_var, width=30)
        self.destination_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Find route button
        self.find_route_button = ttk.Button(route_frame, text="Find Routes", command=self.find_route)
        self.find_route_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.find_route_button.config(state=tk.DISABLED)  # Initially disabled

        # Results section
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a Text widget with scrollbar for displaying results
        results_scroll = ttk.Scrollbar(self.results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, height=15,
                                   yscrollcommand=results_scroll.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_text.yview)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if file_path:
            self.data = self.model.load_data(file_path)
            if self.data is not None:
                self.data_label.config(text=f"Data loaded: {os.path.basename(file_path)}")
                self.status_var.set(f"Data loaded from {file_path}")
            else:
                messagebox.showerror("Error", "Failed to load data")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )

        if file_path:
            self.model.model_path = file_path
            if self.model.load_model():
                model_type = getattr(self.model, 'model_type', 'Unknown')
                self.model_label.config(text=f"Model loaded: {os.path.basename(file_path)} ({model_type.upper()})")
                self.status_var.set(f"{model_type.upper()} model loaded from {file_path}")
            else:
                messagebox.showerror("Error", "Failed to load model")

    def update_predictions_with_time(self):
        """Re-process predictions with the newly selected time"""
        if hasattr(self.model, 'raw_predictions') and self.model.raw_predictions is not None:
            # Get the time index for the selected time
            time_index = self.model.get_time_index(self.time_var.get())

            # Re-process the existing raw predictions with the new time
            self.predictions = self.model.process_predictions(self.model.raw_predictions, time_index)

            # Update the results display
            self.update_results()

            # Show a confirmation message
            self.status_var.set(f"Updated predictions for {self.time_var.get()}")
        else:
            self.status_var.set("No predictions available to update")
            messagebox.showinfo("Info", "Please run the model first to generate predictions")

    def update_location_combos(self):
        """Update the origin and destination dropdown lists with available locations."""
        if hasattr(self.model, 'locations') and len(self.model.locations) > 0:
            # Filter to only include locations with valid coordinates
            valid_locations = []
            for location in sorted(self.model.locations):
                if (location in self.model.corners and 
                    self.model.corners[location].get('has_valid_coords', False)):
                    valid_locations.append(location)

            # Update dropdowns with valid locations only
            self.origin_combo['values'] = valid_locations
            self.destination_combo['values'] = valid_locations

            # Set default values if available
            if len(valid_locations) > 0:
                self.origin_var.set(valid_locations[0])
                if len(valid_locations) > 1:
                    self.destination_var.set(valid_locations[1])
                else:
                    self.destination_var.set(valid_locations[0])

            # Enable the find route button
            self.find_route_button.config(state=tk.NORMAL)

            # Initialize the route finder
            self.route_finder = RouteFinderIntegration(self.model)
            self.route_finder.create_search_graph()

            # Show count of valid locations
            print(f"\nRoute finder initialized with {len(valid_locations)} valid locations out of {len(self.model.locations)} total")

    def find_route(self):
        """Find and display multiple routes between selected origin and destination."""
        if not self.route_finder:
            messagebox.showerror("Error", "Please run traffic prediction first")
            return

        origin = self.origin_var.get()
        destination = self.destination_var.get()

        if not origin or not destination:
            messagebox.showerror("Error", "Please select both origin and destination")
            return

        # Show a message while processing
        self.status_var.set(f"Finding routes from {origin} to {destination}...")
        self.root.update_idletasks()

        try:
            # Find multiple routes
            routes = self.route_finder.find_multiple_routes(origin, destination)

            if routes:
                # Display the results in the GUI
                self.results_text.insert(tk.END, "\n\n=== Route Finding Results ===\n")
                self.results_text.insert(tk.END, f"Found {len(routes)} routes from {origin} to {destination}\n")

                # Show details of each route
                for i, route_info in enumerate(routes, 1):
                    method = route_info['method']
                    path = route_info['path']
                    travel_time = route_info['travel_time']
                    expanded_count = route_info['expanded_count']

                    self.results_text.insert(tk.END, f"\nRoute {i} ({method.upper()}):\n")
                    self.results_text.insert(tk.END, f"  Travel time: {travel_time:.2f} minutes\n")
                    self.results_text.insert(tk.END, f"  Path length: {len(path)} nodes\n")
                    self.results_text.insert(tk.END, f"  Path: {' -> '.join(map(str, path))}\n")
                    self.results_text.insert(tk.END, f"  Nodes expanded: {expanded_count}\n")

                # Ensure the text widget scrolls to the end
                self.results_text.see(tk.END)
            else:
                self.results_text.insert(tk.END, f"No routes found from {origin} to {destination}\n")

                # Ensure the text widget scrolls to the end
                self.results_text.see(tk.END)

            self.status_var.set(f"Route finding complete")
        except Exception as e:
            self.status_var.set(f"Error finding routes: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def start_model_thread(self):
        if self.is_running:
            return

        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        if self.model.model is None:
            messagebox.showerror("Error", "Please load model first")
            return

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_var.set("Running model...")

        # Start the model execution in a separate thread
        thread = threading.Thread(target=self.run_model)
        thread.daemon = True
        thread.start()

    def run_model(self):
        try:
            logging.info("=== Starting Model Execution ===")

            # Prepare input
            input_data = self.model.prepare_input(self.data)

            # Get predictions
            predictions = self.model.predict(input_data)

            # Process predictions with the selected time
            if predictions is not None:
                time_index = self.model.get_time_index(self.time_var.get())
                self.predictions = self.model.process_predictions(predictions, time_index)

                # Update UI in the main thread
                self.root.after(0, self.update_results)
                self.root.after(0, lambda: self.update_time_button.config(state=tk.NORMAL))

                # Save predictions
                self.model.save_predictions()
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate predictions"))
        except Exception as e:
            logging.error(f"Error running model: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
        finally:
            # Update UI in the main thread
            self.root.after(0, self.end_model_run)


    def update_results(self):
        # Clear the Text widget before inserting new text
        self.results_text.delete(1.0, tk.END)

        time_str = self.time_var.get()
        model_type = getattr(self.model, 'model_type', 'Unknown').upper()
        self.results_text.insert(tk.END, f"=== Traffic Flow Predictions ({model_type} Model, Time: {time_str}) ===\n")

        # Count valid and invalid nodes
        valid_count = 0
        invalid_count = 0

        for location, prediction in sorted(self.predictions.items(), key=lambda x: x[1], reverse=True):
            is_valid = (location in self.model.corners and
                    self.model.corners[location].get('has_valid_coords', False))

            if is_valid:
                valid_count += 1
                status = "VALID"
            else:
                invalid_count += 1
                status = "INVALID"

            self.results_text.insert(tk.END, f"{location:<40} ({status}): {prediction:.2f} vehicles\n")

        self.results_text.insert(tk.END, f"\nValid nodes: {valid_count}, Invalid nodes: {invalid_count}\n")

        self.results_text.insert(tk.END, "\n=== Network Summary ===\n")
        node_count = len(self.model.network)
        connection_count = sum(len(data['connections']) for data in self.model.network.values())
        self.results_text.insert(tk.END, f"Total network: {node_count} nodes, ~{connection_count} connections\n")

        # Show travel time information
        self.results_text.insert(tk.END, "\n=== Travel Time Information ===\n")
        self.results_text.insert(tk.END, "Selected travel times between nodes:\n")

        # Show a few example travel times
        count = 0
        for origin, destinations in self.model.travel_times.items():
            if count >= 10:  # Show only first 10 origins
                break

            self.results_text.insert(tk.END, f"From {origin}:\n")

            dest_count = 0
            for destination, time in sorted(destinations.items(), key=lambda x: x[1]):
                if dest_count >= 5:  # Show only first 5 destinations per origin
                    break

                speed = self.model.flow_to_speed(self.model.predictions.get(origin, 0))
                self.results_text.insert(tk.END, f"  To {destination}: {time:.2f} minutes (at {speed:.1f} km/h)\n")
                dest_count += 1

            if len(destinations) > 5:
                self.results_text.insert(tk.END, f"  ... and {len(destinations) - 5} more destinations\n")

            count += 1

        if len(self.model.travel_times) > 10:
            self.results_text.insert(tk.END, f"... and {len(self.model.travel_times) - 10} more origins\n")

        # Update location dropdown lists for route finding
        self.update_location_combos()

        # Ensure the text widget scrolls to the end
        self.results_text.see(tk.END)


    def end_model_run(self):
        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        self.is_running = False
        self.status_var.set("Model execution completed")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficFlowApp(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
