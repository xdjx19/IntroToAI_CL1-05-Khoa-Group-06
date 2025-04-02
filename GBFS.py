import heapq
import math
import sys

class Graph:
    def __init__(self):
        self.nodes = {}  # Stores node positions {id: (x, y)}
        self.edges = {}  # Adjacency list {id: [(neighbor, cost)]}
        self.origin = None
        self.destinations = set()

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = (x, y)
        self.edges[node_id] = []

    def add_edge(self, node1, node2, cost):
        self.edges[node1].append((node2, cost))

    def set_origin(self, origin):
        self.origin = origin

    def set_destinations(self, destinations):
        self.destinations = set(destinations)

    def heuristic(self, node_id, goal_id):
        """Calculate Euclidean distance heuristic."""
        x1, y1 = self.nodes[node_id]
        x2, y2 = self.nodes[goal_id]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def read_graph_from_file(filename):
    """Reads the graph from a text file and returns a Graph object."""
    graph = Graph()
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    section = None  # Tracks whether we are reading Nodes, Edges, Origin, or Destinations

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        if line.startswith("Nodes:"):
            section = "nodes"
        elif line.startswith("Edges:"):
            section = "edges"
        elif line.startswith("Origin:"):
            section = "origin"
        elif line.startswith("Destinations:"):
            section = "destinations"
        else:
            if section == "nodes":
                node_id, coords = line.split(": ")
                x, y = map(int, coords.strip("()").split(","))
                graph.add_node(int(node_id), x, y)
            
            elif section == "edges":
                edge_info, cost = line.split(": ")
                node1, node2 = map(int, edge_info.strip("()").split(","))
                graph.add_edge(node1, node2, int(cost))
            
            elif section == "origin":
                graph.set_origin(int(line))
            
            elif section == "destinations":
                destinations = list(map(int, line.split(";")))
                graph.set_destinations(destinations)
    
    return graph

def gbfs(graph):
    """Greedy Best-First Search implementation."""
    start = graph.origin
    goal = min(graph.destinations, key=lambda g: graph.heuristic(start, g))  # Pick closest goal

    priority_queue = []
    heapq.heappush(priority_queue, (graph.heuristic(start, goal), start, [start]))  # (heuristic, node, path)
    visited = set()

    num_nodes_expanded = 0

    while priority_queue:
        _, current, path = heapq.heappop(priority_queue)

        if current in visited:
            continue
        visited.add(current)
        num_nodes_expanded += 1

        if current in graph.destinations:  # Goal reached
            return current, num_nodes_expanded, path  # Return goal, nodes expanded, and path

        for neighbor, _ in graph.edges[current]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (graph.heuristic(neighbor, goal), neighbor, path + [neighbor]))

    return None, num_nodes_expanded, None  # No path found

# Step 2: Run the Program with Correct Output Format

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()  # Convert to uppercase 

    if method != "GBFS":
        print("Error: Only 'GBFS' method is supported in this version.")
        sys.exit(1)

    graph = read_graph_from_file(filename)
    
    goal, expanded, path = gbfs(graph)

    if path:
        print(f"{filename} {method}")  # First line: filename and method
        print(f"{goal} {expanded}")  # Second line: goal and nodes expanded
        print(" ".join(map(str, path)))  # Third line: path (space-separated)
    else:
        print(f"{filename} {method}")
        print("No path found.")
