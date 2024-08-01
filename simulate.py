from objs import Car, Environment
import tkinter as tk

UPDATE_TIME = 200       # time after which positions are updated (in milliseconds)

def run():
    # Create the main window
    root = tk.Tk()
    root.title("Autonomous car obstacle avoidance")

    # close the program on key press
    def on_key_press(event):
        root.destroy()
        exit(0)
    
    root.bind("<KeyPress>", on_key_press)

    agent = Car()
    env = Environment()

    has_collided = False

    # Set the size of the canvas
    rows = env.GRID_ROWS
    cols = env.GRID_COLS
    
    cell_size = 10  # Size of each cell
    canvas_width = cols * cell_size
    canvas_height = rows * cell_size

    # Create a canvas widget
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    def draw():
        nonlocal has_collided

        # clear the canvas
        canvas.delete("all")

        # Draw the grid
        for i in range(rows):
            for j in range(cols):
                if env.grid[i][j] == 1:
                    x1 = j * cell_size
                    y1 = i * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")
        
        # Draw the car
        car_id = canvas.create_rectangle(
            agent.gridpos()[0] * cell_size,
            agent.gridpos()[1] * cell_size,
            agent.gridpos()[0] * cell_size + cell_size * agent.length,
            agent.gridpos()[1] * cell_size + cell_size * agent.width,
            fill="red",
            outline=""
        )
        
        # check if car if intersecting any obstacle (other than itself)
        
        # if len(canvas.find_overlapping(*canvas.bbox(car_id))) > 1:
        #     has_collided = True
        if env.intersectsWith(agent):
            has_collided = True
    
    def update():
        nonlocal has_collided
        env.shift(1)
        draw()

        if not has_collided:
            root.after(UPDATE_TIME, update)

    draw()
    update()
    root.mainloop()

if __name__ == '__main__':
    run()