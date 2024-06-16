from objs import Car, Environment
import tkinter as tk

def run():
    # Create the main window
    root = tk.Tk()
    root.title("Autonomous car obstacle avoidance")

    agent = Car()
    env = Environment()

    # Set the size of the canvas
    rows = env.GRID_ROWS
    cols = env.GRID_COLS
    
    cell_size = 10  # Size of each cell
    canvas_width = cols * cell_size
    canvas_height = rows * cell_size

    # Create a canvas widget
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Draw the grid
    for i in range(rows):
        for j in range(cols):
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            color = "white" if env.grid[i][j] == 0 else "black"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
    
    # Draw the car
    canvas.create_rectangle(
        agent.gridpos()[0] * cell_size,
        agent.gridpos()[1] * cell_size,
        agent.gridpos()[0] * cell_size + cell_size * agent.length,
        agent.gridpos()[1] * cell_size + cell_size * agent.width,
        fill="red",
        outline=""
    )

    root.mainloop()

if __name__ == '__main__':
    run()