# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def optimum_policy2D(grid,init,goal,cost):
    unknown_value = 9999
    unknown_action = -1
    values = [[[[unknown_value, unknown_action] for row in range(len(grid[0]))] for col in range(len(grid))] for act in range(len(forward))]
    open = [[0, init[0], init[1], init[2], unknown_action]]
    while len(open) > 0:
        # tag value
        current = open.pop()
        g, x, y, d, a = current[0], current[1], current[2], current[3], current[4]
        if values[d][x][y][0] > g:
            values[d][x][y] = [g,a]
        # check possible action paths
        for a2 in range(len(action)):
            d2 = (d + action[a2]) % len(forward)
            x2 = x + forward[d2][0]
            y2 = y + forward[d2][1]
            g2 = g + cost[a2]
            if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0 and values[d2][x2][y2][0] > g2:
                open.append([g2, x2, y2, d2, a2])
    path_available = False
    for d in range(len(forward)):
        path_available = path_available or values[d][goal[0]][goal[1]][0] != unknown_value
    if not path_available:
        return ['fail']
    policy2D = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
    policy2D[goal[0]][goal[1]] = '*'
    current = goal
    tick = unknown_value
    direction = unknown_value
    # find starting point
    for d in range(len(forward)):
        x, y = current[0], current[1]
        if values[d][x][y][0] < tick:
            tick = values[d][x][y][0]
            direction = d
    while current != [init[0],init[1]]:
        x, y = current[0], current[1]
        a = values[direction][x][y][1]
        x2 = x - forward[direction][0]
        y2 = y - forward[direction][1]
        policy2D[x2][y2] = action_name[a]
        current = [x2, y2]
        direction = (direction - action[a]) % len(forward)
    return policy2D

for p in optimum_policy2D(grid,init,goal,cost):
    print(p)
