import numpy as np
import random

INIT_POS = {'x': 20.0, 'y': 25.0}   # initial position of car
TIME_STEP = 0.1
CAR_TO_OBSTACLE_STARTING_GAP = 20
MIN_GAP = 2                         # minimum gap between top and bottom parts of a column
GAP_BETWEEN_COLS = 1                # horizontal gap between two conseutive obstacle columns

class Car:
    width = 1       # kept constant (so that car can be treated as a line in 2D)
    length = 3

    pos = [INIT_POS['x'], INIT_POS['y']]    # x and y are measured from the top left corner
    angle = 0.0                             # clockwise is taken as positive
    
    speed = 1.0

    def __init__(self, length = None) -> None:
        if (length is not None):
            self.length = length

    # returns the x_shift so that the grid can then be shifted left by that amount
    def update_pos(self, steer_angle):
        self.angle += (self.speed * steer_angle * np.pi * TIME_STEP) / (self.length * 180)
        self.pos[1] += (self.speed * np.sin(self.angle) * TIME_STEP)
        return (self.speed * np.cos(self.angle) * TIME_STEP)

    def accelerate(self, acceleration):
        self.speed += acceleration * TIME_STEP
    
    # returns position (x, y) in int
    def gridpos(self):
        return (int(self.pos[0]), int(self.pos[1]))


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x_small, self.x_big = [x1, x2] if (x1 <= x2) else [x2, x1]
        self.y_small, self.y_big = [y1, y2] if (y1 <= y2) else [y2, y1]
    
    # determines whether the line is intersecting a vertical line between (y1, y2)
    def isLineIntersecting(self, x, y1, y2)->bool:
        if (x < self.x_small or x > self.x_big):
            return False
        
        if (abs(self.x_small - self.x_big) < 0.01):
            if (self.y_small > y1 and self.y_big < y2):
                return False
            return True

        slope = float(self.y_big - self.y_small) / (self.x_big - self.x_small)
        y_at_x = self.y_small + slope * (x - self.x_small)
        
        if (y1 < y_at_x < y2):
            return False
        return True


class Environment:
    GRID_ROWS = 50
    GRID_COLS = 100

    prev_gap_bot = 0
    prev_gap_top = GRID_ROWS

    last_obstacle_col_i = 0

    grid = np.zeros((GRID_ROWS, GRID_COLS))     # stores the obstacle grid.
    gridInfo = np.full((2, GRID_COLS), -1)      # stores the lowermost top obstacle index, uppermost bottom obstacle index
                                                # If no obstacle in the column, then both set to -1.
    
    shift_remaining = 0.0               # fractional shift from previous call of shift()

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
            self.gridInfo[0][i], self.gridInfo[1][i] = i_top, i_bot

            for row in range(0, self.GRID_ROWS):
                if (row <= i_top or row >= i_bot):
                    self.grid[row][i] = 1       # obstacle (block)
                else:
                    self.grid[row][i] = 0       # clear path
            
            self.last_obstacle_col_i = i     # set the index of the last obstacle column

            # filling gapped cols with 0s
            curr_i = i + 1
            while (curr_i - i <= GAP_BETWEEN_COLS and curr_i < self.GRID_COLS):
                for row in range(0, self.GRID_ROWS):
                    self.grid[row][curr_i] = 0
                    self.gridInfo[:,curr_i] = -1
                curr_i += 1

            self.prev_gap_bot = gap_bot
            self.prev_gap_top = gap_top

    def shift(self, left):
        totshift = left + self.shift_remaining
        left = int(totshift)
        self.shift_remaining = totshift - left

        if (left == 0):
            return None

        for i in range(0, self.GRID_COLS - left):
            self.grid[:,i] = self.grid[:,i + left]
            self.gridInfo[:,i] = self.gridInfo[:,i + left]

        self.last_obstacle_col_i -= left

        if (self.last_obstacle_col_i + GAP_BETWEEN_COLS + 1 < self.GRID_COLS):
            self.setblocks(startfrom = self.last_obstacle_col_i + GAP_BETWEEN_COLS + 1)
        else:
            for i in range(self.last_obstacle_col_i + 1, self.GRID_COLS):
                self.grid[:,i] = 0
                self.gridInfo[:,i] = -1

    # check collision, and also check if car is inside the bounds of the window
    def intersectsWith(self, agent: Car):
        car_x, car_y = agent.pos

        if int(car_y) < 0 or int(car_x) > (self.GRID_ROWS - 1):
            return True
        
        car_x2 = car_x + agent.length * np.cos(agent.angle)
        car_y2 = car_y + agent.length * np.sin(agent.angle)
        car_line = Line(car_x, car_y, car_x2, car_y2)

        # The car will be treated as a line (since its width is fixed to 1)
        # To find the intersection of a line with an obstacle column, we just need
        # to check a certain simple condition

        for col in range(self.GRID_COLS):
            if ((col > car_x and col > car_x2) or (col < car_x and col < car_x2)):
                continue

            if (self.gridInfo[0][col] == -1):
                continue
            
            i_top, i_bot = self.gridInfo[0][col], self.gridInfo[1][col]

            if (car_line.isLineIntersecting(col, 0, i_top) or \
                car_line.isLineIntersecting(col, i_bot, self.GRID_ROWS - 1)):
                return True

        return False