# Homework 1 - BFS and DFS Implementation

## Overview
This homework contains the implementation of Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms for robot planning on a grid. The assignment involves manually tracing the BFS and DFS algorithms on a 3x3 grid, implementing FIFO and LIFO open sets, and writing a general grid search function that can be used for both BFS and DFS.

## Assignment Details

### Part 1: Tracing BFS and DFS
- **BFS Tracing**: Manually trace the BFS algorithm on a 3x3 grid, listing the order in which nodes are visited, updating the queue, and maintaining the parent dictionary.
- **DFS Tracing**: Similarly, trace the DFS algorithm on the same grid, updating the stack and parent dictionary.
- **Pros and Cons**: Analyze the performance of BFS and DFS on different grid configurations, including scenarios with obstacles.

### Part 2: Implementing BFS and DFS
- **FIFO Queue**: Implement a FIFO queue as an open set for BFS.
- **LIFO Stack**: Implement a LIFO stack as an open set for DFS.
- **Retracing a Plan**: Implement a function to retrace the path from the goal node back to the start node using the parent dictionary.
- **Grid Search Function**: Write a general grid search function that can be used for both BFS and DFS, considering obstacles in the grid.


## Dependencies
- `collections.deque` for FIFO queue implementation
