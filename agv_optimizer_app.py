# agv_optimizer_app.py

import streamlit as st
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Core Pathfinding Logic (A* Algorithm) ---

class Node:
    """Represents a node in the pathfinding grid."""
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current node to end)
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def __lt__(self, other):
        return self.f < other.f

class AStarPathfinder:
    """Implements the A* pathfinding algorithm."""
    def __init__(self, grid, obstacles):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.obstacles = obstacles

    def _is_valid(self, r, c):
        """Checks if a position is within grid bounds and not an obstacle."""
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.obstacles

    def _heuristic(self, a, b):
        """Calculates Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, end):
        """Finds the optimal path from start to end using A*."""
        if not self._is_valid(start[0], start[1]) or not self._is_valid(end[0], end[1]):
            return None, "Start or end position is invalid (out of bounds or an obstacle)."

        start_node = Node(start)
        end_node = Node(end)

        open_list = []
        closed_list = set()

        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1], "Path found successfully!"

            closed_list.add(current_node)

            # Define possible movements (up, down, left, right)
            # (dr, dc) for (delta_row, delta_column)
            movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            for dr, dc in movements:
                neighbor_pos = (current_node.position[0] + dr, current_node.position[1] + dc)

                if self._is_valid(neighbor_pos[0], neighbor_pos[1]):
                    neighbor = Node(neighbor_pos, current_node)

                    if neighbor in closed_list:
                        continue

                    # Cost from start to neighbor
                    neighbor.g = current_node.g + 1
                    # Heuristic cost from neighbor to end
                    neighbor.h = self._heuristic(neighbor.position, end_node.position)
                    # Total cost
                    neighbor.f = neighbor.g + neighbor.h

                    # Check if neighbor is already in open_list with a higher G cost
                    # If so, update it. For simplicity, we'll just add if not present
                    # or if new path is better.
                    found_in_open = False
                    for i, node in enumerate(open_list):
                        if node == neighbor:
                            found_in_open = True
                            if neighbor.g < node.g:
                                open_list[i] = neighbor # Update node in heap (re-heapify might be needed)
                                heapq.heapify(open_list) # Re-heapify after update
                            break
                    if not found_in_open:
                        heapq.heappush(open_list, neighbor)
        return None, "No path found."

# --- 2. Streamlit Application ---

st.set_page_config(layout="wide", page_title="AGV Route Optimizer")

st.title("📦 AGV Route Optimization in E-Commerce Warehouses")
st.markdown("""
This application demonstrates AGV (Automated Guided Vehicle) route optimization
using the A* pathfinding algorithm in a simulated warehouse environment.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Warehouse Configuration")

# Warehouse Dimensions
rows = st.sidebar.slider("Warehouse Rows", 5, 50, 20)
cols = st.sidebar.slider("Warehouse Columns", 5, 50, 20)

# Obstacles Input
st.sidebar.subheader("Obstacles (row,col)")
obstacle_input = st.sidebar.text_area(
    "Enter obstacle coordinates (e.g., 2,3; 5,5; 10,12)",
    value="2,3; 5,5; 10,12"
)
st.sidebar.info("Separate coordinates with semicolons (;). Each coordinate pair should be row,col.")

# AGV Start and End Points
st.sidebar.subheader("AGV Start & End Points")
start_input = st.sidebar.text_input("Start Point (row,col)", value="0,0")
end_input = st.sidebar.text_input("End Point (row,col)", value=f"{rows-1},{cols-1}")

# Parse Inputs
obstacles = set()
try:
    if obstacle_input.strip():
        for obs_str in obstacle_input.split(';'):
            r, c = map(int, obs_str.strip().split(','))
            if 0 <= r < rows and 0 <= c < cols:
                obstacles.add((r, c))
            else:
                st.sidebar.warning(f"Obstacle {obs_str} is out of bounds and will be ignored.")
except Exception:
    st.sidebar.error("Invalid obstacle format. Please use 'row,col; row,col'.")
    obstacles = set() # Reset to empty if parsing fails

start_point = None
end_point = None
try:
    start_r, start_c = map(int, start_input.split(','))
    if 0 <= start_r < rows and 0 <= start_c < cols and (start_r, start_c) not in obstacles:
        start_point = (start_r, start_c)
    else:
        st.sidebar.error("Invalid Start Point: out of bounds or an obstacle.")
except Exception:
    st.sidebar.error("Invalid Start Point format. Please use 'row,col'.")

try:
    end_r, end_c = map(int, end_input.split(','))
    if 0 <= end_r < rows and 0 <= end_c < cols and (end_r, end_c) not in obstacles:
        end_point = (end_r, end_c)
    else:
        st.sidebar.error("Invalid End Point: out of bounds or an obstacle.")
except Exception:
    st.sidebar.error("Invalid End Point format. Please use 'row,col'.")

# --- Main Content Area ---
st.header("Warehouse Visualization & Route")

if st.button("Optimize Route"):
    if start_point and end_point:
        # Create a dummy grid (A* only needs dimensions and obstacles)
        warehouse_grid = np.zeros((rows, cols))

        pathfinder = AStarPathfinder(warehouse_grid, obstacles)
        path, message = pathfinder.find_path(start_point, end_point)

        st.subheader("Optimization Results")
        st.info(message)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis() # Invert Y-axis to match (row, col) indexing where (0,0) is top-left

        # Draw warehouse cells
        for r in range(rows):
            for c in range(cols):
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='lightgray', facecolor='white')
                ax.add_patch(rect)

        # Draw obstacles
        for obs_r, obs_c in obstacles:
            rect = patches.Rectangle((obs_c - 0.5, obs_r - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
            ax.text(obs_c, obs_r, 'X', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw start and end points
        if start_point:
            ax.text(start_point[1], start_point[0], 'S', color='green', ha='center', va='center', fontsize=16, fontweight='bold')
            circle_s = patches.Circle((start_point[1], start_point[0]), 0.4, color='green', alpha=0.5)
            ax.add_patch(circle_s)
        if end_point:
            ax.text(end_point[1], end_point[0], 'E', color='red', ha='center', va='center', fontsize=16, fontweight='bold')
            circle_e = patches.Circle((end_point[1], end_point[0]), 0.4, color='red', alpha=0.5)
            ax.add_patch(circle_e)

        # Draw path
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, color='blue', linewidth=3, marker='o', markersize=8, markerfacecolor='cyan')

        st.pyplot(fig)
    else:
        st.error("Please correct the input errors for Start/End points or Obstacles.")

st.markdown("""
---
### How it Works:
1.  **Warehouse Grid:** The warehouse is represented as a grid where each cell is a potential location for the AGV.
2.  **Obstacles:** You can define specific cells as obstacles that the AGV cannot pass through.
3.  **A* Algorithm:** This is a popular pathfinding algorithm that efficiently finds the shortest path between a start and an end point on a grid. It uses a heuristic function (Manhattan distance in this case) to estimate the cost to the target, guiding the search.
4.  **Visualization:** The optimized path is displayed on the grid, showing the AGV's proposed route.

### Potential Enhancements:
* **Multi-AGV Coordination:** Implement logic to avoid collisions and optimize routes for multiple AGVs simultaneously.
* **Dynamic Obstacles:** Handle moving obstacles or temporary blockages.
* **Real-time Data Integration:** Connect to actual warehouse management systems for live data.
* **Battery Management:** Incorporate battery levels and charging stations into route planning.
* **Traffic Management:** Implement one-way paths or preferred routes.
* **3D Visualization:** Use libraries like `pyvista` or `plotly` for a more immersive 3D view.
""")