import numpy as np

INIT_POS = {'x': 20.0, 'y': 25.0}
TIME_STEP = 0.1
CAR_TO_OBSTACLE_STARTING_GAP = 20

class Car:
    width = 1
    length = 3

    pos = [INIT_POS['x'], INIT_POS['y']]
    angle = 0.0
    
    speed = 1.0
    steer_angle = 0.0

    def __init__(self, length = None) -> None:
        self.length = length

    def update_pos(self):
        self.angle += (self.speed * self.steer_angle * TIME_STEP) / self.length
        self.pos[0] += (self.speed * np.cos(self.angle) * TIME_STEP)
        self.pos[1] += (self.speed * np.sin(self.angle) * TIME_STEP)

    def accelerate(self, acceleration):
        self.speed += acceleration * TIME_STEP
    
    # returns position in grid
    def gridpos(self):
        return [int(self.pos[0]), int(self.pos[1])]


class Environment:
    GRID_ROWS = 50
    GRID_COLS = 100

    grid = np.zeros((GRID_ROWS, GRID_COLS))
    shift_remaining = 0.0 # fractional shift from previous call of shift()

    def __init__(self) -> None:
        self.setblocks(startfrom = INIT_POS['x'] + CAR_TO_OBSTACLE_STARTING_GAP)

    def block(self, x, y):
        self.grid[x][y] = 1

    def setblocks(self, startfrom):
        pass

    def shift(self, left):
        totshift = left + self.shift_remaining
        left = int(totshift)
        self.shift_remaining = totshift - left

        if (left == 0):
            return None

        for i in range(0, self.GRID_COLS - left - 1):
            self.grid[i,:] = self.grid[i + left,:]
        
        self.setblocks(startfrom = self.GRID_COLS - left)