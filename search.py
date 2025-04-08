# --- Required Imports ---
import argparse  # For parsing command-line arguments
import sys  # For system-specific functions like sys.exit
from abc import ABC, abstractmethod  # For creating abstract base classes
import itertools  # Used for IDDFS depth iteration (itertools.count)
from collections import deque
import heapq


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

    def __init__(self, graph, origin, destinations, nodes):
        self.graph = graph
        self.origin = origin
        # Store destinations as a set for efficient membership testing.
        # Checking 'item in my_set' is O(1) on average due to hashing,
        # much faster than O(n) for lists, especially for large numbers
        # of destinations. Also automatically handles duplicate destinations.
        self.destinations = set(destinations)
        self.nodes = nodes
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
import heapq
import math

class GBFS(SearchAlgorithm):
    """Greedy Best-First Search algorithm using a priority queue."""

    def heuristic(self, node, goal, nodes):
        x1, y1 = nodes[node]
        x2, y2 = nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def search(self):
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        visited = set()

        # Use the closest goal as the main target
        goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d, self.nodes))

        # Priority queue: (heuristic value, node, path)
        queue = [(self.heuristic(self.origin, goal, self.nodes), self.origin, [self.origin])]

        while queue:
            _, current, path = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)
            self.expanded_count += 1

            if current in self.destinations:
                self.results[current] = path
                break  # Stop when one goal is reached

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    h = self.heuristic(neighbor, goal, self.nodes)
                    heapq.heappush(queue, (h, neighbor, new_path))

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

            # Stop exploring this path if the depth limit is reached
            if current_depth >= depth_limit:
                continue

            # Add neighbors to the stack if they don't create a cycle in the current path
            # and haven't been visited more shallowly in this DLS run.
            neighbors = self.get_neighbors(node)
            for neighbor in reversed(neighbors):
                # Basic cycle check within the current path
                if neighbor not in path:
                    # Check if visiting this neighbor would be redundant in this DLS
                    if neighbor not in visited_in_dls or visited_in_dls[neighbor] > current_depth + 1:
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path, current_depth + 1))

        return found_new_dest_at_depth

    def search(self):
        """
        Performs the full IDDFS by iteratively calling DLS with increasing depth limits.
        """
        # Reset overall search state
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        self.found_destinations = set()

        # Iterate through depth limits: 0, 1, 2, ...
        for depth_limit in itertools.count():
            # Perform DLS for the current depth limit
            self.depth_limited_search(depth_limit)

            # Stop if all target destinations have been found
            if len(self.found_destinations) == len(self.destinations):
                break

        return self.results, self.expanded_count


# --- Search Method Mapping ---
# Maps command-line method names to their corresponding algorithm classes
SEARCH_METHODS = {
    "dfs": DFS,
    "gbfs": GBFS,
    "iddfs": IDDFS,
    # 'bfs': BFS, # Example for adding Breadth-First Search later
}


# --- Results Display Function ---
def display_results(filename, method, results, expanded_count):
    """
    Prints the search results in the specified format.

    Args:
        filename: The input graph file name.
        method: The search method used (e.g., 'dfs').
        results: The dictionary mapping destinations to paths (or None).
        expanded_count: The total number of nodes expanded.
    """
    found_path = False
    # Sort destinations for consistent output order
    sorted_destinations = sorted(results.keys())

    # Check each destination for a found path
    for dest in sorted_destinations:
        path = results[dest]
        if path:
            # Print results for the first found path (as per apparent requirement)
            print(f"{filename} {method}")
            print(f"{dest} {expanded_count}")
            print(" ".join(map(str, path))) # Join path nodes with spaces
            found_path = True
            break # Exit after printing the first path

    # If no path was found to any destination
    if not found_path:
        print(f"{filename} {method}")
        print(f"None {expanded_count}") # Indicate no path found, still show count


# --- Main Execution Block ---
def main():
    """Parses arguments, runs the selected search algorithm, and displays results."""
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Search a graph file.")
    parser.add_argument("filename", help="Path to the graph file.")
    parser.add_argument(
        "method",
        choices=list(SEARCH_METHODS.keys()), # Allow only defined methods
        help="Search method (e.g., dfs, iddfs)",
    )
    args = parser.parse_args() # Parse the actual arguments provided

    # --- Execute search within a try block for error handling ---
    try:
        # 1. Load graph data from the specified file
        print(f"Loading graph from {args.filename}...")
        graph, origin, destinations, nodes = parse_graph(args.filename)
        print("Graph loaded.")

        # --- Basic Input Validation ---
        if origin is None:
            print("Error: Origin node not specified in the file.")
            sys.exit(1)
        if not destinations:
            print("Error: No destinations specified in the file.")
            sys.exit(1)
        if origin not in graph:
            print(f"Error: Origin node '{origin}' not found in graph nodes/edges.")
            sys.exit(1)
        # (Could add checks for destinations existing in graph too)

        # 2. Select the appropriate search algorithm class
        print(f"Selected search method: {args.method}")
        search_class = SEARCH_METHODS[args.method]

        # 3. Instantiate the search algorithm object
        search_algorithm = search_class(graph, origin, destinations, nodes)
        print("Search algorithm created.")

        # 4. Run the search
        print("Starting search...")
        results, expanded_count = search_algorithm.search()
        print("Search finished.")

        # 5. Display the results
        display_results(args.filename, args.method, results, expanded_count)

    # --- Handle potential errors during execution ---
    except Exception as e:
        # Catch any unexpected errors during parsing or search
        print(f"\n--- An Error Occurred ---")
        print(f"Error details: {e}")
        print("Please check the file format and command line arguments.")
        sys.exit(1) # Exit with an error code

#BFS and UCS Code (Dwayne D'Souza)


def read_graph(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    nodes_section = edges_section = False
    nodes = {}
    edges = {}
    origin = None
    destinations = []

    for line in lines:
        if line.startswith("Nodes:"):
            nodes_section = True
            edges_section = False
            continue
        elif line.startswith("Edges:"):
            nodes_section = False
            edges_section = True
            continue
        elif line.startswith("Origin:"):
            origin = int(line.split(":")[1].strip())
        elif line.startswith("Destinations:"):
            destinations = list(map(int, line.split(":")[1].strip().split(";")))
        elif nodes_section and ':' in line:
            parts = line.split(":")
            node = int(parts[0].strip())
            coord = tuple(map(int, parts[1].strip(" ()").split(",")))
            nodes[node] = coord
        elif edges_section and ':' in line:
            parts = line.split(":")
            edge_nodes = tuple(map(int, parts[0].strip("()").split(",")))
            cost = int(parts[1].strip())
            if edge_nodes[0] not in edges:
                edges[edge_nodes[0]] = []
            edges[edge_nodes[0]].append((edge_nodes[1], cost))

    return nodes, edges, origin, destinations

def bfs(edges, start, goals):
    queue = deque([[start]])
    visited = set()
    nodes_created = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_created += 1

        if node in goals:
            return node, nodes_created, path

        if node not in visited:
            visited.add(node)
            for neighbor, _ in sorted(edges.get(node, []), key=lambda x: x[0]):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

    return None, nodes_created, []

def ucs(edges, start, goals):
    frontier = [(0, [start])]  # (cost, path)
    visited = {}
    nodes_created = 0

    while frontier:
        cost, path = heapq.heappop(frontier)
        node = path[-1]
        nodes_created += 1

        if node in goals:
            return node, nodes_created, path

        if node not in visited or cost < visited[node]:
            visited[node] = cost
            for neighbor, edge_cost in sorted(edges.get(node, []), key=lambda x: x[0]):
                new_cost = cost + edge_cost
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(frontier, (new_cost, new_path))

    return None, nodes_created, []

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename = sys.argv[1]
    method = sys.argv[2].lower()

    nodes, edges, origin, destinations = read_graph(filename)

    if method == "bfs":
        goal, count, path = bfs(edges, origin, destinations)
    elif method == "ucs":
        goal, count, path = ucs(edges, origin, destinations)
    else:
        print("Only 'bfs' and 'ucs' methods are supported in this version.")
        return

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")

# --- Script Entry Point ---
# Ensures main() is called only when the script is executed directly
if __name__ == "__main__":
    main()
