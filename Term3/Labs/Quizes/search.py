# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    current = [0, init[0], init[1]]
    open_spots = [current]
    grid[current[1]][current[2]] = 2
    path = "fail"
    while len(open_spots) > 0:
        # 1. find smallest g-value item from the list of open spots, select it
        min_idx = 0
        for i in range(len(open_spots)):
            if (open_spots[i][0] < open_spots[min_idx][0]):
                min_idx = i
        current = open_spots[min_idx]
        open_spots.pop(min_idx)
        # 2. if the selected item is the goal, set path and break loop
        if current[1] == goal[0] and current[2] == goal[1]:
            # goal reached
            path = current
            break
        # 3. expand neighbors of the selected item, add to open spots list
        if current[1] > 0:
            # expansion to the left is allowed
            if grid[current[1]-1][current[2]] == 0:
                open_spots.append([current[0]+1, current[1]-1, current[2]])
                grid[current[1]-1][current[2]] = 2
        if current[2] > 0:
            # expansion upwards is allowed
            if grid[current[1]][current[2]-1] == 0:
                open_spots.append([current[0]+1, current[1], current[2]-1])
                grid[current[1]][current[2]-1] = 2
        if current[1] < len(grid)-1:
            # expansion to the right is allowed
            if grid[current[1]+1][current[2]] == 0:
                open_spots.append([current[0]+1, current[1]+1, current[2]])
                grid[current[1]+1][current[2]] = 2
        if current[2] < len(grid[0])-1:
            # expansion downwards is allowed
            if grid[current[1]][current[2]+1] == 0:
                open_spots.append([current[0]+1, current[1], current[2]+1])
                grid[current[1]][current[2]+1] = 2
    return path

print(search(grid, init, goal, cost))
