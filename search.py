# --- Required Imports ---
import argparse  # For parsing command-line arguments
import sys  # For system-specific functions like sys.exit
from abc import ABC, abstractmethod  # For creating abstract base classes
import itertools  # Used for IDDFS depth iteration (itertools.count)
from collections import deque #Used for BFS to queue to store multiple tuples
import matplotlib.pyplot as plt
import networkx as nx
import os

def visualize_path(graph, nodes, path, filename, method):

    try:

        # Create a networkx graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        for node_id, pos in nodes.items():
            G.add_node(node_id, pos=pos)
        
        # Add edges
        for node_id, neighbors in graph.items():
            for neighbor, _ in neighbors:  # Ignoring edge weights
                G.add_edge(node_id, neighbor)
        
        # Get positions for all nodes
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw the base graph
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Highlight the path if it exists
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=500, node_color='salmon')
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='red', arrows=True)
            
            # Highlight start and end nodes
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=600, node_color='green')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=600, node_color='purple')
        
        # Add node labels (just the node IDs)
        nx.draw_networkx_labels(G, pos)
        
        # Remove axis
        plt.axis('off')
        plt.title(f"Path using {method.upper()} - {os.path.basename(filename)}")
        
        # Save the figure
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_{method}_path.png"
        plt.savefig(output_filename, format="PNG", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved as {output_filename}")
        
    except ImportError:
        print("Unable to create visualization. Make sure matplotlib and networkx are installed.")
        print("You can install them with: pip install matplotlib networkx")


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

    def search(self):
        """
        Performs the full IDDFS by iteratively calling DLS with increasing depth limits.
        Terminates when all destinations are found or when no new nodes can be explored.
        """
        # Reset overall search state
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        self.found_destinations = set()
        
        # Track all reachable nodes
        all_visited_nodes = set()
        prev_visited_count = 0
        
        # Iterate through depth limits: 0, 1, 2, ...
        for depth_limit in itertools.count():
            # Perform DLS for the current depth limit
            visited_this_depth = set()
            self.depth_limited_search(depth_limit, visited_this_depth)
            
            # Add newly visited nodes to our total set
            all_visited_nodes.update(visited_this_depth)
            
            # Stop if all target destinations have been found
            if len(self.found_destinations) == len(self.destinations):
                break
                
            # If we didn't discover any new nodes at this depth limit,
            # we've explored the entire reachable graph and can stop
            if len(all_visited_nodes) == prev_visited_count:
                break
                
            prev_visited_count = len(all_visited_nodes)

        return self.results, self.expanded_count

    def depth_limited_search(self, depth_limit, visited_nodes):
        """
        Performs a Depth-Limited Search (DLS) up to a specified depth.

        Args:
            depth_limit: The maximum depth to explore in this iteration.
            visited_nodes: Set to track all nodes visited in this DLS.

        Returns:
            bool: True if any new destination was found at this depth, False otherwise.
        """
        found_new_dest_at_depth = False
        # Stack stores: (node, current_depth)
        stack = [(self.origin, 0)]
        # Track the path to the current node
        path = {self.origin: [self.origin]}
        
        # Track visited nodes with the depth they were first visited at
        visited_in_dls = {self.origin: 0}
        
        # Add origin to visited nodes set
        visited_nodes.add(self.origin)
        
        while stack:
            node, current_depth = stack.pop()
            current_path = path[node]
            
            # Check if this is a destination we haven't found yet
            if node in self.destinations and node not in self.found_destinations:
                self.results[node] = current_path
                self.found_destinations.add(node)
                found_new_dest_at_depth = True
            
            # Count this node as expanded
            self.expanded_count += 1
            
            # Stop exploring this path if we've reached the depth limit
            if current_depth >= depth_limit:
                continue
                    
            # Process neighbors
            neighbors = self.get_neighbors(node)
            for neighbor in reversed(neighbors):
                new_depth = current_depth + 1
                
                # Skip if adding this neighbor would create a cycle in the current path
                if neighbor in current_path:
                    continue
                    
                # Skip if we've already visited this neighbor at a shallower or equal depth
                if neighbor in visited_in_dls and visited_in_dls[neighbor] <= new_depth:
                    continue
                    
                # Record the depth for this neighbor
                visited_in_dls[neighbor] = new_depth
                
                # Add to visited nodes set
                visited_nodes.add(neighbor)
                
                # Create the path to this neighbor
                new_path = current_path + [neighbor]
                path[neighbor] = new_path
                
                # Add to stack
                stack.append((neighbor, new_depth))

        return found_new_dest_at_depth

    
    # --- Breadth-First Search Implementation --- (Dwayne D'Souza)
class BFS(SearchAlgorithm):
    """Implements Breadth-First Search using a queue."""

    def search(self):
        # Reset search-specific state
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        self.found_destinations = set()

        # Use a queue for BFS: stores tuples (node, path_list)
        # Initialize with the origin node and its path (just the origin itself)
        queue = deque([(self.origin, [self.origin])])
        # Track visited nodes to avoid cycles
        visited = set([self.origin])

        while queue:
            current_node, path = queue.popleft()
            self.expanded_count += 1

            # Check if current node is an unfound destination
            if current_node in self.destinations and current_node not in self.found_destinations:
                self.results[current_node] = path
                self.found_destinations.add(current_node)
                # Optimization: Stop if all destinations are found
                if self.found_destinations == self.destinations:
                    break

            # Add neighbors to the queue
            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return self.results, self.expanded_count

class AStar(SearchAlgorithm):
    """Implements A* search using a priority queue and Euclidean distance heuristic."""
    
    def __init__(self, graph, origin, destinations, nodes):
        super().__init__(graph, origin, destinations, nodes)  # Fixed: properly pass nodes parameter
        
    def heuristic(self, node):
        """Euclidean distance to the closest destination."""
        if not self.nodes or node not in self.nodes:
            return 0  # Return 0 if coordinates not available
            
        min_dist = float('inf')  # Initialize to a very large number (infinity)
        x1, y1 = self.nodes[node]  # Get the coordinates of the current node
        
        for dest in self.destinations:  # Loop through all destinations
            if dest in self.nodes:  # Make sure the destination has coordinates
                x2, y2 = self.nodes[dest]  # Get the coordinates of the destination
                dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5  # Calculate the Euclidean distance
                if dist < min_dist:  # Update min_dist if we found a closer destination
                    min_dist = dist
                    
        return min_dist

    def search(self):
        # Reset the search state
        self.expanded_count = 0  # Number of nodes expanded
        self.results = {dest: None for dest in self.destinations}  # Initialize the results
        self.found_destinations = set()  # Set to track found destinations

        # Priority queue: (est_total_cost, cost_so_far, node, path, node_id_for_tiebreaker)
        open_set = []
        # Start with the origin node, its heuristic value, 0 cost, and just the origin in the path
        heapq.heappush(open_set, (self.heuristic(self.origin), 0, self.origin, [self.origin], self.origin))
        
        # Track visited nodes and their costs
        visited = {}  # {node: best_cost_so_far}

        while open_set:
            est_total, cost_so_far, current, path, _ = heapq.heappop(open_set)

            # Skip if we've found a better path to this node already
            if current in visited and visited[current] <= cost_so_far:
                continue
                
            # Record this as the best path to the current node
            visited[current] = cost_so_far
            self.expanded_count += 1

            # Check if this is a destination we haven't found yet
            if current in self.destinations and current not in self.found_destinations:
                self.results[current] = path
                self.found_destinations.add(current)
                # If we've found all destinations, we can stop
                if self.found_destinations == self.destinations:
                    break

            # Explore neighbors
            for neighbor, edge_cost in self.graph.get(current, []):
                new_cost = cost_so_far + edge_cost
                # Only consider if we haven't visited or found a better path
                if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                    est_cost = new_cost + self.heuristic(neighbor)
                    # Add to priority queue with all necessary info
                    heapq.heappush(
                        open_set,
                        (est_cost, new_cost, neighbor, path + [neighbor], neighbor)
                    )

        return self.results, self.expanded_count

# --- Uniform Cost Search Implementation --- (Dwayne D'Souza)
class UCS(SearchAlgorithm):
    """Implements Uniform Cost Search using a priority queue."""

    def search(self):
        # Reset search-specific state
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        self.found_destinations = set()

        # Priority queue: (total_cost, node, path_list)
        # Initialize with the origin node (cost 0) and its path
        heap = [(0, self.origin, [self.origin])]
        # Track visited nodes and their minimum costs
        visited = {}

        while heap:
            current_cost, current_node, path = heapq.heappop(heap)
            self.expanded_count += 1

            # Skip if we've found a better path to this node already
            if current_node in visited and visited[current_node] < current_cost:
                continue

            # Check if current node is an unfound destination
            if current_node in self.destinations and current_node not in self.found_destinations:
                self.results[current_node] = path
                self.found_destinations.add(current_node)
                # Optimization: Stop if all destinations are found
                if self.found_destinations == self.destinations:
                    break

            # Add neighbors to the priority queue
            for neighbor, edge_cost in self.graph.get(current_node, []):
                new_cost = current_cost + edge_cost
                # Only proceed if this is a better path than previously found
                if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                    visited[neighbor] = new_cost
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_cost, neighbor, new_path))

        return self.results, self.expanded_count

# --- Search Method Mapping ---
# Maps command-line method names to their corresponding algorithm classes
SEARCH_METHODS = {
    "dfs": DFS,
    "gbfs": GBFS,
    "iddfs": IDDFS,
    "bfs": BFS,
    "ucs": UCS,
    "astar": AStar,
}


def display_results(filename, method, results, expanded_count, graph, nodes):
    """
    Prints the search results in the specified format and generates a visualization.
    """
    found_path = False
    # Sort destinations for consistent output order
    sorted_destinations = sorted(results.keys())

    # Check each destination for a found path
    for dest in sorted_destinations:
        path = results[dest]
        if path:
            # Print results for the first found path
            print(f"{filename} {method}")
            print(f"{dest} {expanded_count}")
            print(" ".join(map(str, path))) # Join path nodes with spaces
            
            # Create a visualization of the path
            visualize_path(graph, nodes, path, filename, method)
            
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
        display_results(args.filename, args.method, results, expanded_count, graph, nodes)

    # --- Handle potential errors during execution ---
    except Exception as e:
        # Catch any unexpected errors during parsing or search
        print(f"\n--- An Error Occurred ---")
        print(f"Error details: {e}")
        print("Please check the file format and command line arguments.")
        sys.exit(1) # Exit with an error code

# --- Script Entry Point ---
# Ensures main() is called only when the script is executed directly
if __name__ == "__main__":
    main()
