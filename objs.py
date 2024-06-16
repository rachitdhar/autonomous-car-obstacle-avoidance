import numpy as np
import random

INIT_POS = {'x': 20.0, 'y': 25.0}
TIME_STEP = 0.1
CAR_TO_OBSTACLE_STARTING_GAP = 20
MIN_GAP = 2
GAP_BETWEEN_COLS = 1

class Car:
    width = 1
    length = 3

    pos = [INIT_POS['x'], INIT_POS['y']]
    angle = 0.0
    
    speed = 1.0
    steer_angle = 0.0

    def __init__(self, length = None) -> None:
        if (length is not None):
            self.length = length

    def update_pos(self):
        self.angle += (self.speed * self.steer_angle * TIME_STEP) / self.length
        self.pos[0] += (self.speed * np.cos(self.angle) * TIME_STEP)
        self.pos[1] += (self.speed * np.sin(self.angle) * TIME_STEP)

    def accelerate(self, acceleration):
        self.speed += acceleration * TIME_STEP
    
    # returns position (x, y) in int
    def gridpos(self):
        return (int(self.pos[0]), int(self.pos[1]))


class Environment:
    GRID_ROWS = 50
    GRID_COLS = 100

    prev_gap_bot = 0
    prev_gap_top = GRID_ROWS

    grid = np.zeros((GRID_ROWS, GRID_COLS))
    shift_remaining = 0.0 # fractional shift from previous call of shift()

    def __init__(self) -> None:
        self.setblocks(startfrom = int(INIT_POS['x'] + CAR_TO_OBSTACLE_STARTING_GAP))

    def setblocks(self, startfrom):
        for i in range(startfrom, self.GRID_COLS, GAP_BETWEEN_COLS + 1):
            gap_bot = self.GRID_ROWS    # pos of gap top
            gap_top = 0                 # pos of gap bottom
            
            while (gap_bot + MIN_GAP >= self.prev_gap_top):
                gap_bot = random.randint(0, self.GRID_ROWS - MIN_GAP)
            
            while (gap_top - MIN_GAP <= self.prev_gap_bot):
                gap_top = random.randint(gap_bot + MIN_GAP, self.GRID_ROWS)
            
            i_top = self.GRID_ROWS - gap_top - 1        # index of the lowermost top obstacle
            i_bot = self.GRID_ROWS - gap_bot            # index of the uppermost bottom obstacle

            for row in range(0, self.GRID_ROWS):
                if (row <= i_top or row >= i_bot):
                    self.grid[row][i] = 1       # obstacle (block)
                else:
                    self.grid[row][i] = 0       # clear path
            
            # filling gapped cols with 0s
            curr_i = i + 1
            while (curr_i - i <= GAP_BETWEEN_COLS and curr_i < self.GRID_COLS):
                for row in range(0, self.GRID_ROWS):
                    self.grid[row][curr_i] = 0
                curr_i += 1

            self.prev_gap_bot = gap_bot
            self.prev_gap_top = gap_top

    def shift(self, left):
        totshift = left + self.shift_remaining
        left = int(totshift)
        self.shift_remaining = totshift - left

        if (left == 0):
            return None

        for i in range(0, self.GRID_COLS - left - 1):
            self.grid[i,:] = self.grid[i + left,:]
        
        self.setblocks(startfrom = self.GRID_COLS - left)