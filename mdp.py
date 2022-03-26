# imports
import json
import copy
import argparse
import random

# Visualise the map
def printEnvironment(map, row, col, reward, whitespace_reward, policy=False, init=False):
    p_map = ""
    for r in range(row):
        p_map += "|"
        for c in range(col):
            val = ''
            if map[r][c] == 99:
                val = 'WALL'
            elif policy:
                val = ["Down", "Left", "Up", "Right"][map[r][c]]
            elif map[r][c] != 0:
                val = str(map[r][c])
                if init:
                    reward[r][c] = map[r][c]
                    map[r][c] = 0
            else:
                val = str(map[r][c])
                if init:
                    reward[r][c] = whitespace_reward
                    map[r][c] = 0
            p_map += f' {val.ljust(5)} |'
        p_map += '\n'
    print(p_map)
    # print('reward Map')
    # for r in reward:
    #     print(r)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, row, col, actions, action):
    dr, dc = actions[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= row or newC >= col or (U[newR][newC] == 99): # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, reward, discount, row,col, actions, action):
    u = reward[r][c]
    u += 0.1 * discount * getU(U, r, c, row, col, actions, (action-1) % 4)
    u += 0.8 * discount * getU(U, r, c, row, col, actions, action)
    u += 0.1 * discount * getU(U, r, c, row, col, actions, (action+1) % 4)
    return u

# method for value iteration
def valueIteration(U, row, col, num_actions, max_error, discount, reward, actions, whitespace_reward):
    iteration = 0
    original = copy.deepcopy(U)
    while True:
        iteration+=1
        nextU = copy.deepcopy(original)
        error = 0
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99:
                    continue
                # bellman equation
                nextU[r][c] = max([calculateU(U, r, c, reward, discount, row, col, actions, action) for action in range(num_actions)])
                error = max(error, abs(nextU[r][c] - U[r][c]))
        U = nextU
        printEnvironment(U, row, col, reward, whitespace_reward)
        if error < max_error * (1-discount) / discount:
            break
    print(f'Total iteration {iteration}')
    return U

# Get the optimal policy from U
def getOptimalPolicy(U, row, col, num_actions, reward, discount, actions):
    policy = [[-1 for j in range(col)] for i in range(row)]
    for r in range(row):
        for c in range(col):
            if U[r][c] == 99:
                policy[r][c] = 99
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(num_actions):
                u = calculateU(U, r, c,reward, discount, row, col, actions, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy

# Perform some simplified value iteration steps to get an approximation of the utilities
def policyEvaluation(policy, U, reward, discount, row, col, actions, max_error):
    original = copy.deepcopy(U)
    while True:
        nextU = copy.deepcopy(original)
        error = 0
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99:
                    policy[r][c] = 99
                    continue
                nextU[r][c] = calculateU(U, r, c, reward, discount, row, col, actions, policy[r][c]) # simplified Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        if error < max_error * (1-discount) / discount:
            break
    return U

def policyIteration(policy, U, reward, discount, row, col, actions, num_actions, max_error, whitespace_reward):
    iteration = 0
    while True:
        iteration+=1
        U = policyEvaluation(policy, U, reward, discount, row, col, actions, max_error)
        unchanged = True
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99:
                    policy[r][c] = 99
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(num_actions):
                    u = calculateU(U, r, c, reward, discount, row, col, actions, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculateU(U, r, c, reward, discount, row, col, actions, policy[r][c]):
                    policy[r][c] = maxAction # the action that maximizes the utility
                    unchanged = False
        if unchanged:
            break
        printEnvironment(policy, row, col, reward, whitespace_reward)
    print(f'Total iteration {iteration}')
    return policy


# main method
def main():
    num_actions = 4
    actions = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
    reward = None
    U = []

    # Get arguments
    parser = argparse.ArgumentParser(description='Markov Decision Process, a grid world example involving value and policy iteration.')
    parser.add_argument('--discount', type=float, default=0.99, help='discount for bellman equation.')
    parser.add_argument('--max_error', type=float, default=62, help='Max error for epsilon value.')
    parser.add_argument('--whitespace_reward', type=float, default=-0.04, help='Rewards for the white tiles.')
    parser.add_argument('--row', type=float, default=6, help='Number of rows in map.')
    parser.add_argument('--col', type=float, default=6, help='Number of columns in map.')
    parser.add_argument('--map', type=str, default='./map', help='Filepath to map. Must be in JSON format.')
    parser.add_argument('--algorithm', type=int, default=0, help='Choose \n 0: for value iteration \n 1: policy iteration')
    
    args = parser.parse_args()
    discount = args.discount
    max_error = args.max_error
    whitespace_reward = args.whitespace_reward
    row = args.row
    col = args.col
    map = args.map
    algorithm = args.algorithm

    # Print the initial environment
    # Load map and reward
    with open(f'{map}.json', 'r') as f:
        U = json.load(f)
        if len(U) > 0:
            row = len(U)
            col = len(U[0])
            reward = [[0 for r in U[0]] for c in U]
        else:
            print('Invalid map!')
        # reward = [[0 for i in range(len(U)+1)]for j in range(len(U)+1)]
    print("The initial U is:\n")
    printEnvironment(U, row, col, reward, whitespace_reward, init=True)

    if algorithm == 0:
        print("Value iteration selected \n")
        # Value iteration
        U = valueIteration(U, row, col, num_actions, max_error, discount, reward, actions, whitespace_reward)

        # Get the optimal policy from U and print it
        policy = getOptimalPolicy(U, row, col, num_actions, reward, discount, actions)
        print("The optimal policy is:")
        printEnvironment(policy, row, col, reward, whitespace_reward, True)
    elif algorithm == 1:
        print("Policy iteration selected \n")
        policy = [[random.randint(0, 3) for j in range(col)] for i in range(row)]

        # Policy iteration
        policy = policyIteration(policy, U, reward, discount, row, col, actions, num_actions, max_error, whitespace_reward)

        # Print the optimal policy
        print("The optimal policy is:\n")
        printEnvironment(policy, row, col, reward, whitespace_reward, True)

# Program entry point
if __name__ == "__main__":
    main()