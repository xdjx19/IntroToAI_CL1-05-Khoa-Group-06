import sys
from collections import deque

def parse_input_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = set()

    with open(filename, 'r') as file:
        lines = file.read().splitlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.endswith(":"):
            section = line[:-1]
            continue
        if section == "Nodes":
            node_id, coord = line.split(":")
            nodes[int(node_id.strip())] = eval(coord.strip())
        elif section == "Edges":
            edge, cost = line.split(":")
            edge = edge.strip()[1:-1]  # remove parentheses
            from_node, to_node = map(int, edge.split(","))
            cost = int(cost.strip())
            if from_node not in edges:
                edges[from_node] = []
            edges[from_node].append((to_node, cost))
        elif section == "Origin":
            origin = int(line.strip())
        elif section == "Destinations":
            destinations = set(map(int, line.split(";")))

    return nodes, edges, origin, destinations


def bfs_search(nodes, edges, origin, destinations):
    visited = set()
    queue = deque([[origin]])
    num_nodes = 0

    while queue:
        path = queue.popleft()
        current = path[-1]
        num_nodes += 1

        if current in destinations:
            return current, num_nodes, path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(edges.get(current, [])):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
    return None, num_nodes, []


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> bfs")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].lower()

    if method != "bfs":
        print("Only 'bfs' method is supported in this script.")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_input_file(filename)
    goal, num_nodes, path = bfs_search(nodes, edges, origin, destinations)

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {num_nodes}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")
