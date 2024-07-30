import tkinter as tk
import numpy as np

class Renderer:
    def __init__(self, env, car):
        self.env = env
        self.car = car
        self.root = None
        self.canvas = None
        self.cell_size = 10
        self.canvas_width = env.GRID_COLS * self.cell_size
        self.canvas_height = env.GRID_ROWS * self.cell_size

    def render(self, mode='human'):
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Autonomous car obstacle avoidance")
            self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
            self.canvas.pack()

        self.canvas.delete("all")

        # Draw the grid
        for i in range(self.env.GRID_ROWS):
            for j in range(self.env.GRID_COLS):
                if self.env.grid[i][j] == 1:
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")

        # Draw the car
        car_x, car_y = self.car.pos
        car_length = self.car.length * self.cell_size
        car_width = self.car.width * self.cell_size

        # Calculate the corners of the car
        corners = [
            (-car_length/2, -car_width/2),
            (car_length/2, -car_width/2),
            (car_length/2, car_width/2),
            (-car_length/2, car_width/2)
        ]

        # Rotate the corners
        rotated_corners = []
        for x, y in corners:
            rx = x * np.cos(self.car.angle) - y * np.sin(self.car.angle)
            ry = x * np.sin(self.car.angle) + y * np.cos(self.car.angle)
            rotated_corners.append((rx + car_x * self.cell_size, ry + car_y * self.cell_size))

        # Draw the rotated car
        self.canvas.create_polygon(rotated_corners, fill="red", outline="")

        self.root.update()

    def close(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
