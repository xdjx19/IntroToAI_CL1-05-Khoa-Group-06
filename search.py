class DFS(SearchAlgorithm):
    """Depth-First Search algorithm using a standard Python list as a stack."""

    # Renamed this method from run_search to search
    def search(self):
        """
        Perform Depth-First Search on the graph using a list as a stack.

        Returns:
            dict: Paths to target points (or None if unreachable)
            int: Number of nodes processed
        """
        # --- Use variables from the base class __init__ ---
        self.expanded_count = 0 # Changed from nodes_processed_count
        # Reset results using the base class attribute name
        self.results = {dest: None for dest in self.destinations} # Changed from paths_to_goals and target_points
        # Use the base class attribute for found destinations
        self.found_destinations = set() # Changed from goals_reached

        seen_nodes = set() # Local variable for visited tracking in this search

        # Use a list as a stack: stores tuples (node, path_tuple)
        # Path is stored as a tuple internally for slight variation
        # Use base class attribute 'origin'
        nodes_to_visit = [(self.origin, (self.origin,))] # Changed from start_point

        while nodes_to_visit:
            # Get next item to explore (LIFO behavior with pop())
            current_node, path_tuple = nodes_to_visit.pop()

            # Skip nodes already fully processed
            if current_node in seen_nodes:
                continue

            # Mark as seen and count using the base class attribute
            seen_nodes.add(current_node)
            self.expanded_count += 1 # Changed from nodes_processed_count

            # Check if we found a target point using base class attributes
            if current_node in self.destinations and current_node not in self.found_destinations:
                # Convert path tuple back to list for the result, store in base class attribute
                self.results[current_node] = list(path_tuple) # Changed from paths_to_goals
                self.found_destinations.add(current_node) # Changed from goals_reached
                # Optional: Stop early if all targets are found
                # Use base class attributes for comparison
                if self.found_destinations == self.destinations: # Changed from goals_reached == self.target_points
                    break

            # Add adjacent nodes to the list acting as a stack
            # Use base class method 'get_neighbors'
            adjacent_nodes = self.get_neighbors(current_node) # Changed from retrieve_neighbors
            for next_node in reversed(adjacent_nodes):
                if next_node not in seen_nodes:
                    # Create the extended path tuple
                    extended_path_tuple = path_tuple + (next_node,)
                    nodes_to_visit.append((next_node, extended_path_tuple))

        # Return the base class attributes
        return self.results, self.expanded_count # Changed from paths_to_goals, nodes_processed_count


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


