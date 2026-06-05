import matplotlib.pyplot as plt
import cv2 
import numpy as np

class State:
    """Represents a state (point) in the search space."""
    
    def __init__(self, x, y, z=None):
        """Initializes the state with coordinates (x, y) and an optional z value (elevation)."""
        self.x = x 
        self.y = y
        self.z = z

    def get_attribute(self):
        """Returns a list of the state's attributes [x, y, z]."""
        return [self.x, self.y, self.z]
    
    def get_value(self):
        """Returns the z value (used as the objective function value to evaluate)."""
        return self.z
    
    def print_state(self):
        """Prints the state information to the console."""
        print(f"State({self.x}, {self.y}, {self.z})")

class Problem:
    """Defines the state space problem based on 2D image data."""

    def __init__(self, image_path):
        """Initializes the problem and preprocesses the image data."""
        self.image_path = image_path
        self.X, self.Y, self.Z = self._load()   

    def _load(self):
        """Reads the image, downscales it (25%), applies Gaussian blur, and caches the result."""
        img = cv2.imread(self.image_path, 0)
        if img is None:
            raise ValueError("img is None")
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        return np.arange(w), np.arange(h), img

    def load_state_space(self):
        """Returns the cached state space arrays (X, Y, Z)."""
        return self.X, self.Y, self.Z           

    def get_initial_state(self):
        """Gets the default initial state at coordinates (0, 0)."""
        z = self.Z[0][0]
        return State(0, 0, z)

    def get_state_list(self):
        """Generates and returns a list of all available states in the space."""
        state_space = []
        for y in range(self.Z.shape[0]):
            for x in range(self.Z.shape[1]):
                state_space.append(State(x, y, self.Z[y, x]))
        return state_space

    def get_neighbors(self, state):
        """Gets a list of valid neighboring states (up to 8 directions) for the current state."""
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),           (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        h, w = self.Z.shape                     
        neighbors = []
        for dx, dy in directions:
            nx, ny = state.x + dx, state.y + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbors.append(State(nx, ny, self.Z[ny, nx]))
        return neighbors

    def visualize_space(self, path, best_state, schedule_func):
        """Plots a 3D surface chart of the state space and the search path."""
        X, Y = np.meshgrid(self.X, self.Y)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.Z,
                        rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none', alpha=0.2)
        self.draw_search_path(ax, path, best_state)

        plt.tight_layout()

        if schedule_func == "sd1":
            plt.savefig('figure_sd1.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('figure_sd2.png', dpi=300, bbox_inches='tight')

        plt.show()

    def draw_search_path(self, ax, path, best_state):
        """Draws the algorithm's path and marks the start and best states."""
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        zs = [p.z for p in path]
        ax.plot(xs, ys, zs, color='r', zorder=3, linewidth=0.5)
        self.draw_state(ax, best_state, "Best")
        self.draw_state(ax, self.get_initial_state(), "Start")

    def draw_state(self, ax, state, state_name):
        """Marks a specific state on the 3D plot with a text label."""
        ax.scatter(state.x, state.y, state.z, color='black', s=40)
        ax.text(state.x, state.y, state.z,
                f"{state_name}\n({state.x}, {state.y}, {state.z})")