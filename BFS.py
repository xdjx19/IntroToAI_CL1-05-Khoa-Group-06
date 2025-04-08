import sys
from collections import deque

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

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> bfs")
        return

    filename = sys.argv[1]
    method = sys.argv[2].lower()

    if method != "bfs":
        print("Only 'bfs' method is supported in this version.")
        return

    nodes, edges, origin, destinations = read_graph(filename)
    goal, count, path = bfs(edges, origin, destinations)

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {count}")
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
