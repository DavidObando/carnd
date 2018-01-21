# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

#grid = [[0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0]]
#grid = [[0, 0, 1, 0, 0, 0],
#        [0, 0, 1, 0, 0, 0],
#        [0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0],
#        [0, 0, 1, 1, 1, 0],
#        [0, 0, 0, 0, 1, 0]]
grid = [[0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid,goal,cost):
    not_visited = -1
    barrier = 99
    value = [[not_visited for y in range(len(grid[0]))] for x in range(len(grid))]
    open = [[0, goal[0], goal[1]]]
    while len(open) > 0:
        current = open.pop()
        weight, x, y = current[0], current[1], current[2]
        next_weight = weight+1
        if grid[x][y] == 0:
            value[x][y] = weight
        else:
            value[x][y] = barrier
            continue
        if x > 0:
            # allowed to go left
            next = value[x-1][y]
            if next == not_visited or next > next_weight:
                open.append([next_weight, x-1, y])
        if y > 0:
            # allowed to go up
            next = value[x][y-1]
            if next == not_visited or next > next_weight:
                open.append([next_weight, x, y-1])
        if x < len(grid)-1:
            # allowed to go right
            next = value[x+1][y]
            if next == not_visited or next > next_weight:
                open.append([next_weight, x+1, y])
        if y < len(grid[0])-1:
            # allowed to go down
            next = value[x][y+1]
            if next == not_visited or next > next_weight:
                open.append([next_weight, x, y+1])
    # replace all not_visited nodes for barriers
    for i in range(len(value)):
        for j in range(len(value[0])):
            if (value[i][j] == not_visited):
                value[i][j] = barrier
    return value

result = compute_value(grid,goal,cost)
for l in result:
  print(l)
