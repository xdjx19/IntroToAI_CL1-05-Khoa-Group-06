import argparse
import sys
from abc import ABC, abstractmethod


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
    graph = {}      # Will store {node_id: [(neighbor1, cost1), (neighbor2, cost2), ...]}
    nodes = {}      # Will store {node_id: (x_coord, y_coord)}
    origin = None   # Will store the starting node ID
    destinations = []  # Will store list of target node IDs
    
    # Read the file line by line
    current_section = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
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
                    destinations = [int(d.strip()) for d in line.split(";") if d.strip()]
    
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
        self.destinations = destinations
        self.expanded_count = 0
        self.results = {dest: None for dest in destinations}
    
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
    """Depth-First Search algorithm."""
    
    def search(self):
        """
        Perform Depth-First Search on the graph.
        
        Returns:
            dict: A dictionary mapping each destination to its path from origin (or None if unreachable)
            int: The number of nodes expanded during the search
        """
        # Reset counters and results
        self.expanded_count = 0
        self.results = {dest: None for dest in self.destinations}
        
        # Track visited nodes to avoid cycles
        visited = set()
        
        # Track the current path
        path = []
        
        def dfs_recursive(node):
            # Count this node as expanded
            self.expanded_count += 1
            
            # Mark node as visited and add to current path
            visited.add(node)
            path.append(node)
            
            # Check if this node is a destination
            if node in self.destinations:
                # Store the path to this destination
                self.results[node] = path.copy()
                return True  # Found a destination
            
            # Explore neighbors in sorted order
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    if dfs_recursive(neighbor):
                        return True  # Propagate the success up
            
            # Backtrack: remove node from path when we're done exploring it
            path.pop()
            visited.remove(node)  # Allow node to be visited in different paths
            return False  # No destinations found in this branch
        
        # Start DFS from the origin node
        dfs_recursive(self.origin)
        
        # Return results and node count
        return self.results, self.expanded_count


# Dictionary mapping search method names to their classes
SEARCH_METHODS = {
    'dfs': DFS,
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
    # For each destination that was reached
    for dest, path in results.items():
        if path:
            print(f"{filename} {method}")
            print(f"{dest} {expanded_count}")
            print(" ".join(map(str, path)))
            # We only need to output the first destination reached
            break
    else:
        # If no destination was reached
        print(f"{filename} {method}")
        print("No path found")


def main():
    """Main program function - parses graph and handles command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph search program')
    parser.add_argument('filename', help='Graph file path') #handles the first argument when search.py is called which is the filename of the graph txt
    parser.add_argument('method', choices=list(SEARCH_METHODS.keys()), help='Search method') #handles the first argument when search.py is called which is the search algo the user wants to run
    args = parser.parse_args()
    
    try:
        # Parse graph by calling the parse_graph function with the filename argument inputed by the user
        graph, origin, destinations, nodes = parse_graph(args.filename)
        
        # Get the appropriate search class based on the method inputed by the user
        search_class = SEARCH_METHODS.get(args.method)
        
        if search_class:
            # Create an instance of the search algorithm
            search_algorithm = search_class(graph, origin, destinations)
            
            # Run the selected search algorithm
            results, expanded_count = search_algorithm.search()
            
            # Display the results
            display_results(args.filename, args.method, results, expanded_count)
        else:
            print(f"Error: Method '{args.method}' is not implemented")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()