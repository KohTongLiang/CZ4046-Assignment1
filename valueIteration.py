# imports
import json
import argparse

# Visualise the map
def printEnvironment(map, NUM_ROW, NUM_COL, REWARD, WHITESPACE_REWARD, policy=False):
    p_map = ""
    for r in range(NUM_ROW):
        p_map += "|"
        for c in range(NUM_COL):
            val = ''
            if map[r][c] == 99:
                val = 'WALL'
            elif policy:
                test  = map[r][c]
                val = ["Down", "Left", "Up", "Right"][map[r][c]]
                print()
            elif map[r][c] != 0:
                val = str(map[r][c])
                if policy is False: REWARD[r][c] = map[r][c]
            else:
                val = str(map[r][c])
                if policy is False: REWARD[r][c] = WHITESPACE_REWARD
            p_map += f' {val.ljust(5)} |'
        p_map += '\n'
    print(p_map)
    print('Reward Map')
    for r in REWARD:
        print(r)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (U[newR][newC] == 99): # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, REWARD, DISCOUNT, NUM_ROW,NUM_COL, ACTIONS, action):
    u = REWARD[r][c]
    u += 0.1 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS,  (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS,(action+1)%4)
    return u

# method for value iteration
def valueIteration(U, NUM_ROW, NUM_COL, NUM_ACTIONS, MAX_ERROR, DISCOUNT, REWARD, ACTIONS, WHITESPACE_REWARD):
    print("Value Iteration:\n")
    while True:
        nextU = U.copy()
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if U[r][c] == 99:
                    continue
                # bellman equation
                nextU[r][c] = max([calculateU(U, r, c, REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS, action) for action in range(NUM_ACTIONS)])
                error = max(error, abs(nextU[r][c] - U[r][c]))
        U = nextU
        printEnvironment(U, NUM_ROW, NUM_COL, REWARD, WHITESPACE_REWARD)
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

# Get the optimal policy from U
def getOptimalPolicy(U, NUM_ROW, NUM_COL, NUM_ACTIONS, REWARD, DISCOUNT, ACTIONS):
    policy = [[-1 for j in range(NUM_COL)] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if U[r][c] == 99:
                policy[r][c] = 99
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c,REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


# main method
def main():
    DISCOUNT = 0.99
    MAX_ERROR = 10**(-3)
    NUM_ACTIONS = 4
    ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
    WHITESPACE_REWARD = -0.2
    NUM_ROW = None
    NUM_COL = None
    REWARD = None
    U = []
    # Get arguments
    parser = argparse.ArgumentParser(description='Markov Decision Process, a grid world example.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    args = parser.parse_args()

    # Print the initial environment
    # Load map and reward
    with open('./test_map.json', 'r') as f:
        U = json.load(f)
        if len(U) > 0:
            NUM_ROW = len(U)
            NUM_COL = len(U[0])
        else:
            print('Invalid map!')
        REWARD = [[0 for i in range(len(U)+1)]for j in range(len(U)+1)]
    print("The initial U is:\n")
    printEnvironment(U, NUM_ROW, NUM_COL, REWARD, WHITESPACE_REWARD)

    # Value iteration
    U = valueIteration(U, NUM_ROW, NUM_COL, NUM_ACTIONS, MAX_ERROR, DISCOUNT, REWARD, ACTIONS, WHITESPACE_REWARD)

    # Get the optimal policy from U and print it
    policy = getOptimalPolicy(U, NUM_ROW, NUM_COL, NUM_ACTIONS, REWARD, DISCOUNT, ACTIONS)
    print("The optimal policy is:\n")
    printEnvironment(policy, NUM_ROW, NUM_COL, REWARD, WHITESPACE_REWARD, True)

# Program entry point
if __name__ == "__main__":
    main()