# --- Required Imports ---
import argparse  # For parsing command-line arguments
import sys  # For system-specific functions like sys.exit
from abc import ABC, abstractmethod  # For creating abstract base classes
import itertools  # Used for IDDFS depth iteration (itertools.count)


# --- Graph Parsing Function ---
def parse_graph(filename):
    """Reads a graph file and returns its components."""
    # Initialize data structures to store graph information
    graph = {}  # Adjacency list: {node_id: [(neighbor1, cost1), ...]}
    nodes = {}  # Node coordinates: {node_id: (x_coord, y_coord)}
    origin = None  # Starting node ID
    destinations = []  # List of target node IDs

    current_section = None  # Tracks the current section being parsed (Nodes, Edges, etc.)
    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Identify section headers
                if line == "Nodes:":
                    current_section = "nodes"
                    continue
                elif line == "Edges:":
                    current_section = "edges"
                    continue
                elif line == "Origin:":
                    current_section = "origin"
                    continue
                elif line == "Destinations:":
                    current_section = "destinations"
                    continue

                # Process lines based on the current section
                if current_section == "nodes":
                    # Example line: "1: (4,1)"
                    if ":" in line:
                        node_id_part, coord_part = line.split(":", 1)
                        node_id = int(node_id_part.strip())
                        coord_text = coord_part.strip()[1:-1]  # Remove parentheses
                        x, y = map(int, coord_text.split(","))
                        nodes[node_id] = (x, y)  # Store coordinates
                        graph[node_id] = []  # Initialize empty adjacency list

                elif current_section == "edges":
                    # Example line: "(2,1): 4"
                    if ":" in line:
                        edge_part, cost_part = line.split(":", 1)
                        nodes_text = edge_part.strip()[1:-1]  # Remove parentheses
                        from_node, to_node = map(int, nodes_text.split(","))
                        cost = int(cost_part.strip())

                        # Add edge to the graph (ensure 'from_node' exists)
                        if from_node not in graph:
                            graph[from_node] = []
                        graph[from_node].append((to_node, cost))

                        # Ensure 'to_node' exists in the graph keys for consistency
                        if to_node not in graph:
                            graph[to_node] = []

                elif current_section == "origin":
                    # The line after "Origin:" contains the node ID
                    if line:
                        origin = int(line)

                elif current_section == "destinations":
                    # The line after "Destinations:" contains semicolon-separated IDs
                    if line:
                        # Parse potentially multiple destinations
                        destinations = [
                            int(d.strip())
                            for d in line.split(";")
                            if d.strip()
                        ]

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        sys.exit(1)

    return graph, origin, destinations, nodes


# --- Abstract Base Class for Search Algorithms ---
class SearchAlgorithm(ABC):
    """Defines the common interface for all search algorithms."""

    def __init__(self, graph, origin, destinations):
        self.graph = graph
        self.origin = origin
        # Store destinations as a set for efficient membership testing.
        # Checking 'item in my_set' is O(1) on average due to hashing,
        # much faster than O(n) for lists, especially for large numbers
        # of destinations. Also automatically handles duplicate destinations.
        self.destinations = set(destinations)
        self.expanded_count = 0  # Counter for nodes expanded during search
        # Stores the path found for each destination (initially None)
        self.results = {dest: None for dest in destinations}
        # Tracks which destinations have been successfully reached
        self.found_destinations = set()

    @abstractmethod
    def search(self):
        """
        Executes the specific search algorithm. Must be implemented by subclasses.

        Returns:
            tuple: (results_dict, expanded_count)
                   results_dict maps destination IDs to their paths (list of nodes) or None.
                   expanded_count is the total number of nodes expanded.
        """
        pass  # Subclasses must provide their own implementation

    def get_neighbors(self, node):
        """
        Retrieves neighbors of a given node, sorted by node ID.

        Args:
            node: The ID of the node whose neighbors are requested.

        Returns:
            list: A sorted list of neighbor node IDs.
        """
        # self.graph.get(node, []) returns the list of (neighbor, cost) tuples
        # or an empty list if the node has no neighbors or isn't in the graph.
        neighbor_ids = [neighbor for neighbor, _ in self.graph.get(node, [])]
        return sorted(neighbor_ids)


# --- Depth-First Search Implementation ---
class DFS(SearchAlgorithm):
    """Implements Depth-First Search using a list as a stack."""

    def search(self):
        # Reset search-specific state
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        self.found_destinations = set()

        # Track nodes visited during this specific search to avoid cycles
        seen_nodes = set()

        # Use a list as a LIFO stack: stores tuples (node, path_tuple)
        # Start with the origin node and its path (just the origin itself)
        stack = [(self.origin, (self.origin,))]

        while stack:
            # Get the last added item (LIFO behavior)
            current_node, path_tuple = stack.pop()

            # Skip if this node has already been fully processed
            if current_node in seen_nodes:
                continue

            # Mark as processed and count expansion
            seen_nodes.add(current_node)
            self.expanded_count += 1

            # Check if this node is an unfound destination
            if current_node in self.destinations and current_node not in self.found_destinations:
                self.results[current_node] = list(path_tuple) # Store path
                self.found_destinations.add(current_node)
                # Optimization: Stop if all destinations are found
                if self.found_destinations == self.destinations:
                    break

            # Add neighbors to the stack
            neighbors = self.get_neighbors(current_node)
            # Add neighbors in reverse sorted order so they are popped in sorted order
            for next_node in reversed(neighbors):
                if next_node not in seen_nodes:
                    extended_path = path_tuple + (next_node,)
                    stack.append((next_node, extended_path))

        return self.results, self.expanded_count


# --- Iterative Deepening Depth-First Search Implementation ---
class IDDFS(SearchAlgorithm):
    """Implements Iterative Deepening Depth-First Search."""

    def depth_limited_search(self, depth_limit):
        """
        Performs a Depth-Limited Search (DLS) up to a specified depth.

        Args:
            depth_limit: The maximum depth to explore in this iteration.

        Returns:
            bool: True if any new destination was found at this depth, False otherwise.
        """
        found_new_dest_at_depth = False
        # Stack stores: (node, path_list, current_depth)
        stack = [(self.origin, [self.origin], 0)]
        # Tracks visited nodes *at specific depths* within this DLS iteration
        # Helps avoid cycles and redundant exploration within this depth limit.
        # Format: {node_id: shallowest_depth_visited_at_in_this_dls}
        visited_in_dls = {}

        while stack:
            node, path, current_depth = stack.pop()

            # Pruning based on visits within this DLS:
            # Skip if we've already visited this node at a shallower or equal depth
            # during this specific DLS run.
            if node in visited_in_dls and visited_in_dls[node] <= current_depth:
                continue
            visited_in_dls[node] = current_depth # Record visit at this depth

            # Count expansion (note: IDDFS re-expands nodes at different depths)
            self.expanded_count += 1

            # Check if it's a destination we haven't found a path for yet
            if node in self.destinations and node not in self.found_destinations:
                self.results[node] = path
                self.found_destinations.add(node)
                found_new_dest_at_depth = True
                # Continue DLS: other destinations might be at the same depth

            # Stop exploring this path if the depth limit 
