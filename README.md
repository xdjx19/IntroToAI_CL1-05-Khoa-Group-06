# IntroToAI_CL1-05-Khoa-Group-06

## Assignment 2

This repository contains the implementation of various graph search algorithms.

## Dependencies

Install the required packages using pip:

```bash
pip install networkx matplotlib
```

## Usage

Run the search algorithm using the following syntax:

python search.py "graph.txt" dfs

Where:
- The first argument is the path to the graph file
- The second argument is the search algorithm to use (options: dfs, bfs, iddfs, gbfs, ucs, astar)

## Features

- Supports multiple search algorithms:
  - Depth-First Search (DFS)
  - Breadth-First Search (BFS)
  - Iterative Deepening DFS (IDDFS)
  - Greedy Best-First Search (GBFS)
  - Uniform Cost Search (UCS)
  - A* Search
- Visualizes the found paths using matplotlib and networkx
- Handles weighted directed graphs
- Finds paths from a single origin to one or more destinations

## Graph File Format

The input graph file should be structured as follows:

```
Nodes:
1: (x1,y1)
2: (x2,y2)
...

Edges:
(1,2): cost
(2,3): cost
...

Origin:
1

Destinations:
2;3;4
```

## Output

The program outputs:
1. The filename and search method
2. The found destination and number of expanded nodes
3. The path from origin to destination
4. The path of the graph visualization png
