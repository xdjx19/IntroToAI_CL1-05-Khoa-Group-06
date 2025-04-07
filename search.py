import argparse
import sys
from abc import ABC, abstractmethod
import itertools # Added for IDDFS depth iteration


def parse_graph(filename):
    """
    Parse a graph from a text file for the pathfinding assignment.

    The file should have this format:
    - Nodes section: Lists nodes with their coordinates
    - Edges section: Lists directed edges with their costs
    - Origin section: Specifies the starting node
    - Destinations section: Lists the target nodes to reach

    Returns:
        graph: Dictionary where keys are node IDs and values are lists of (neighbor, cost) tuples
        origin: Starting node ID
        destinations: List of target node IDs
        nodes: Dictionary of node coordinates
    """
    # Initialize data structures
    graph = {}  # Will store {node_id: [(neighbor1, cost1), (neighbor2, cost2), ...]}
    nodes = {}  # Will store {node_id: (x_coord, y_coord)}
    origin = None  # Will store the starting node ID
    destinations = []  # Will store list of target node IDs

    # Read the file line by line
    current_section = None
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Check for section headers
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

            # Process the line based on current section
            if current_section == "nodes":
                # Process a node line: "1: (4,1)"
                if ":" in line:
                    node_id_part, coord_part = line.split(":", 1)
                    node_id = int(node_id_part.strip())

                    # Extract coordinates from parentheses
                    coord_text = coord_part.strip()[1:-1]  # Remove ( and )
                    x, y = map(int, coord_text.split(","))

                    # Store node coordinates
                    nodes[node_id] = (x, y)

                    # Initialize empty adjacency list for this node
                    graph[node_id] = []

            elif current_section == "edges":
                # Process an edge line: "(2,1): 4"
                if ":" in line:
                    edge_part, cost_part = line.split(":", 1)

                    # Extract nodes from parentheses
                    nodes_text = edge_part.strip()[1:-1]  # Remove ( and )
                    from_node, to_node = map(int, nodes_text.split(","))

                    # Extract cost
                    cost = int(cost_part.strip())

                    # Add edge to graph
                    if from_node not in graph:
                        graph[from_node] = []
                    graph[from_node].append((to_node, cost))

                    # Ensure to_node exists in graph
                    if to_node not in graph:
                        graph[to_node] = []

            elif current_section == "origin":
                # Origin node is on a separate line after "Origin:"
                if line:
                    origin = int(line)

            elif current_section == "destinations":
                # Destinations are on a separate line after "Destinations:"
                if line:
                    destinations = [
                        int(d.strip()) for d in line.split(";") if d.strip()
                    ]

    return graph, origin, destinations, nodes


class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms."""

    def __init__(self, graph, origin, destinations):
        """
        Initialize the search algorithm.

        Args:
            graph: Adjacency list representation of the graph with costs
            origin: Starting node
            destinations: List of destination nodes to find
        """
        self.graph = graph
        self.origin = origin
        # Store destinations as a set for faster lookups
        self.destinations = set(destinations)
        self.expanded_count = 0
        # Initialize results with None for all original destinations
        self.results = {dest: None for dest in destinations}
        self.found_destinations = set() # Track which destinations we've found

    @abstractmethod
    def search(self):
        """
        Perform the search algorithm.

        Returns:
            dict: A dictionary mapping each destination to its path from origin (or None if unreachable)
            int: The number of nodes expanded during the search
        """
        pass

    def get_neighbors(self, node):
        """
        Get the neighbors of a node sorted by node ID.

        Args:
            node: The node to get neighbors for

        Returns:
            list: Sorted list of neighbor node IDs
        """
        # Extract neighbor IDs from (neighbor, cost) tuples and sort
        return sorted([neighbor for neighbor, _ in self.graph.get(node, [])])


class DFS(SearchAlgorithm):
    """Depth-First Search algorithm using a standard Python list as a stack."""

    def run_search(self):
        """
        Perform Depth-First Search on the graph using a list as a stack.

        Returns:
            dict: Paths to target points (or None if unreachable)
            int: Number of nodes processed
        """
        # Initialize tracking variables
        self.nodes_processed_count = 0
        # Reset results for this search run
        self.paths_to_goals = {goal: None for goal in self.paths_to_goals}
        goals_reached = set()
        seen_nodes = set()

        # Use a list as a stack: stores tuples (node, path_tuple)
        # Path is stored as a tuple internally for slight variation
        nodes_to_visit = [(self.start_point, (self.start_point,))] # Path as tuple

        while nodes_to_visit:
            # Get next item to explore (LIFO behavior with pop())
            current_node, path_tuple = nodes_to_visit.pop()

            # Skip nodes already fully processed
            if current_node in seen_nodes:
                continue

            # Mark as seen and count
            seen_nodes.add(current_node)
            self.nodes_processed_count += 1

            # Check if we found a target point
            if current_node in self.target_points and current_node not in goals_reached:
                # Convert path tuple back to list for the result
                self.paths_to_goals[current_node] = list(path_tuple)
                goals_reached.add(current_node)
                # Optional: Stop early if all targets are found
                # if goals_reached == self.target_points:
                #     break

            # Add adjacent nodes to the list acting as a stack
            # Note: retrieve_neighbors sorts ascending, so reverse for stack LIFO
            adjacent_nodes = self.retrieve_neighbors(current_node)
            for next_node in reversed(adjacent_nodes):
                if next_node not in seen_nodes:
                    # Create the extended path tuple
                    extended_path_tuple = path_tuple + (next_node,)
                    nodes_to_visit.append((next_node, extended_path_tuple))

        return self.paths_to_goals, self.nodes_processed_count


class IDDFS(SearchAlgorithm):
    """Iterative Deepening Depth-First Search algorithm."""

    def depth_limited_search(self, depth_limit):
        """
        Performs a Depth-Limited Search (DLS) up to a given depth.

        Args:
            depth_limit: The maximum depth to explore.

        Returns:
            bool: True if a goal was found at this depth, False otherwise.
        """
        found_at_this_depth = False
        # Stack holds: (node, path_so_far, current_depth)
        stack = [(self.origin, [self.origin], 0)]
        # Visited set for the *current* DLS iteration to handle cycles within this depth
        visited_in_dls = {} # node -> depth visited at

        while stack:
            node, path, current_depth = stack.pop()

            # Check if we've visited this node at a shallower or equal depth in this DLS
            if node in visited_in_dls and visited_in_dls[node] <= current_depth:
                 continue
            visited_in_dls[node] = current_depth

            # Increment expanded count only when processing a node
            # Note: IDDFS re-expands nodes at different depths
            self.expanded_count += 1

            # Check if we found a destination *that we haven't found yet*
            if node in self.destinations and node not in self.found_destinations:
                self.results[node] = path
                self.found_destinations.add(node)
                found_at_this_depth = True
                # Don't stop DLS here, other destinations might be at the same depth

            # Stop exploring this path if depth limit reached
            if current_depth >= depth_limit:
                continue

            # Add neighbors to stack (in reverse sorted order for correct DFS expansion)
            neighbors = self.get_neighbors(node)
            for neighbor in reversed(neighbors):
                 # Avoid cycles within the current path for this DLS
                 # Also check visited_in_dls to prune branches explored deeper
                 if neighbor not in path:
                    if neighbor not in visited_in_dls or visited_in_dls[neighbor] > current_depth + 1:
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path, current_depth + 1))

        return found_at_this_depth


    def search(self):
        """
        Perform Iterative Deepening Depth-First Search.

        Returns:
            dict: Paths to destinations (or None if unreachable)
            int: Total number of nodes expanded across all iterations
        """
        self.expanded_count = 0
        # Reset results and found destinations for this search run
        self.results = {dest: None for dest in self.results}
        self.found_destinations = set()

        # Iterate through increasing depth limits
        # itertools.count() generates 0, 1, 2, ...
        for depth_limit in itertools.count():
            # Perform DLS for the current depth limit
            found_goal = self.depth_limited_search(depth_limit)

            # If we found all destinations, we can stop
            if len(self.found_destinations) == len(self.destinations):
                 break


        return self.results, self.expanded_count


# Dictionary mapping search method names to their classes
SEARCH_METHODS = {
    "dfs": DFS,
    "iddfs": IDDFS,  # Added IDDFS
    #'bfs': BFS,
    # More methods can be added here
}


def display_results(filename, method, results, expanded_count):
    """
    Display search results in the required format.

    Args:
        filename: The input file name
        method: The search method used
        results: Dictionary mapping destinations to paths
        expanded_count: Number of nodes expanded during search
    """
    found_path = False
    # Iterate through destinations in sorted order for consistent output
    sorted_destinations = sorted(results.keys())

    for dest in sorted_destinations:
        path = results[dest]
        if path:
            print(f"{filename} {method}")
            print(f"{dest} {expanded_count}")
            print(" ".join(map(str, path)))
            found_path = True
            # We only need to output the first destination reached (as per original logic)
            break # Exit after printing the first found path

    if not found_path:
        # If no destination was reached
        print(f"{filename} {method}")
        # Print expanded count even if no path is found
        print(f"None {expanded_count}") # Modified to include expanded_count


def main():
    """Runs the graph search."""
    # --- Get input from the command line ---
    # Set up tool to read command line arguments
    parser = argparse.ArgumentParser(description="Search a graph file.")
    # Tell it we need a filename
    parser.add_argument("filename", help="The graph file.")
    # Tell it we need a search method (like 'dfs' or 'iddfs')
    parser.add_argument(
        "method", choices=list(SEARCH_METHODS.keys()), help="Search method (e.g., dfs)"
    )
    # Read the arguments provided by the user
    args = parser.parse_args()

    # --- Try to do the search ---
    # Use a try block in case something goes wrong (like file not found)
    try:
        # Step 1: Read the graph from the file
        print(f"Loading graph from {args.filename}...")
        graph, origin, destinations, nodes = parse_graph(args.filename)
        print("Graph loaded.")

        # Basic check: Did we find an origin node in the file?
        if origin is None:
            print("Error: Could not find the 'Origin:' node in the file.")
            sys.exit(1) # Stop the program

        # Basic check: Did we find any destination nodes?
        if not destinations:
            print("Error: Could not find any 'Destinations:' in the file.")
            sys.exit(1) # Stop the program

        # Basic check: Does the origin node actually exist in the graph?
        # (Maybe the file listed an origin node that doesn't have any edges or coordinates)
        if origin not in graph:
             print(f"Error: The origin node '{origin}' is listed but not defined in 'Nodes:' or 'Edges:'.")
             sys.exit(1) # Stop the program

        # (Skipping complex checks for whether destinations exist - assume file is mostly okay)

        # Step 2: Figure out which search function to use
        print(f"Selected search method: {args.method}")
        # Look up the class (like DFS or IDDFS) based on the user's input method
        search_class = SEARCH_METHODS[args.method] # Get the class directly

        # Step 3: Create the search object
        # Give the search class the graph details it needs
        search_algorithm = search_class(graph, origin, destinations)
        print("Search algorithm created.")

        # Step 4: Run the search!
        print("Starting search...")
        results, expanded_count = search_algorithm.search()
        print("Search finished.")

        # Step 5: Show the results
        display_results(args.filename, args.method, results, expanded_count)

    # --- Handle errors ---
    # If anything in the 'try' block failed, this 'except' block will run
    except Exception as e:
        # Print a general error message
        print(f"\n--- An Error Occurred ---")
        print(f"Error details: {e}")
        print("Please check the file format and command line arguments.")
        sys.exit(1) # Stop the program because of the error


if __name__ == "__main__":
    main()
