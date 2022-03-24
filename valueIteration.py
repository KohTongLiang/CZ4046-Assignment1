# Arguments
REWARD = -0.04 # constant reward for non-terminal states
DISCOUNT = 0.99
MAX_ERROR = 10**(-3)

# Set up the initial environment
NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
NUM_ROW = 6
NUM_COL = 6
# U = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
U = [[1, 99, 1, 0, 0, 1], 
     [0, -1, 0, 1, 99, 0], 
     [0, 0, -1, 0, 1, 0], 
     [0, 0, 0, -1, 0, 1], 
     [0, 99, 99, 99, -1, 0], 
     [0, 0, 0, 0, 0, 0]]

# Visualization
def printEnvironment(map, policy=False):
    # res = ""
    # for r in range(NUM_ROW):
        # res += "|"
        # for c in range(NUM_COL):
            # if (r == 0 and c == 1) or ):
            #     val = "WALL"
            # if r <= 1 and c == 3:
            #     val = "+1" if r == 0 else "-1"
            # else:
            #     if policy:
            #         val = ["Down", "Left", "Up", "Right"][arr[r][c]]
            #     else:
            #         val = str(arr[r][c])
            # res += " " + val[:5].ljust(5) + " |" # format
        # res += "\n"
    p_map = ""
    for r in range(NUM_ROW):
        p_map += "|"
        for c in range(NUM_COL):
            val = ''
            if map[r][c] == 99:
                val = 'wall'
            elif policy:
                val = ["Down", "Left", "Up", "Right"][map[r][c]]
            else:
                val = str(map[r][c])
            p_map += f' {val.ljust(5)} |'
        p_map += '\n'
    print(p_map)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (U[newR][newC] == 99): # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

# method for value iteration
def valueIteration(U):
    print("During the value iteration:\n")
    while True:
        # nextU = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        nextU = [[1, 99, 1, 0, 0, 1], 
            [0, -1, 0, 1, 99, 0], 
            [0, 0, -1, 0, 1, 0], 
            [0, 0, 0, -1, 0, 1], 
            [0, 99, 99, 99, -1, 0], 
            [0, 0, 0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                # if (r <= 1 and c == 3) or (r == c == 1):
                if U[r][c] == 99:
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c] - U[r][c]))
        U = nextU
        printEnvironment(U)
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

# Get the optimal policy from U
def getOptimalPolicy(U):
    policy = [[-1 for j in range(NUM_COL)] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if U[r][c] == 99:
            # if (r <= 1 and c == 3) or (r == c == 1):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy

# Print the initial environment
print("The initial U is:\n")
printEnvironment(U)

# Value iteration
U = valueIteration(U)

# Get the optimal policy from U and print it
policy = getOptimalPolicy(U)
print("The optimal policy is:\n")
printEnvironment(policy, True)
