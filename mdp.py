# imports
import json
import copy
import argparse
import random
import matplotlib.pyplot as plt

#### Common functions
# Print the environment in a readable manner
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
            p_map += f' {val.ljust(20)} |'
        p_map += '\n'
    print(p_map)


# Given an action, get utility of the next state reached
def getU(U, r, c, row, col, actions, action):
    # get row and col index of next state
    add_r, add_c = actions[action]
    next_r, next_c = r + add_r, c + add_c

    # check if collide with the boundary or the wall
    if next_r < 0 or next_c < 0 or next_r >= row or next_c >= col or (U[next_r][next_c] == 99):
        return U[r][c] # return original if hit boundary or wall
    else:
        return U[next_r][next_c] # return next state

# Given an action, calculate the utlility
def calculateU(U, r, c, reward, discount, row,col, actions, action):
    u = reward[r][c]
    # transition probability x discount factor x utility given an action
    u += 0.1 * discount * getU(U, r, c, row, col, actions, (action-1) % 4)
    u += 0.8 * discount * getU(U, r, c, row, col, actions, action)
    u += 0.1 * discount * getU(U, r, c, row, col, actions, (action+1) % 4)
    return u
#### End of Common functions

#### Solution: Value Iteration
def valueIteration(U, row, col, num_actions, threshold, discount, reward, actions, whitespace_reward, plotter):
    iteration = 0
    original = copy.deepcopy(U)
    while True:
        nextU = copy.deepcopy(original)
        change = 0
        # iterate all states
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99: continue # skip walls
                # Bellman update
                nextU[r][c] = max([calculateU(U, r, c, reward, discount, row, col, actions, action) for action in range(num_actions)])
                change = max(change, abs(nextU[r][c] - U[r][c]))
                plotter[f'{r}-{c}'].append(nextU[r][c]) # for plotting
        # U <-- U'
        U = nextU
        printEnvironment(U, row, col, reward, whitespace_reward)
        iteration+=1
        # if change < threshold stop value iteration
        if change < threshold * (1 - discount) / discount:
            break
    print(f'Total iteration {iteration}')
    return U, iteration

# Get the optimal policy from U after value iterations completed
def getOptimalPolicy(U, row, col, num_actions, reward, discount, actions):
    policy = [[-1 for j in range(col)] for i in range(row)] # initialise policy
    # iterate all states
    for r in range(row):
        for c in range(col):
            if U[r][c] == 99: # skipp wall
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
#### End of Solution: Value Iteration

#### Solution: Policy Iteration
# Perform some simplified value iteration steps to get an approximation of the utilities
def policyEvaluation(policy, U, reward, discount, row, col, actions, threshold, k):
    original = copy.deepcopy(U)
    i = 0
    # k iterations for modified policy iteration
    while i < k:
        nextU = copy.deepcopy(original)
        change = 0
        # iterate all states
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99: # skipp wall
                    policy[r][c] = 99
                    continue
                # Bellman update without max operator
                nextU[r][c] = calculateU(U, r, c, reward, discount, row, col, actions, policy[r][c])
                change = max(change, abs(nextU[r][c]-U[r][c]))
        U = nextU
        i += 1
    return U

def policyIteration(policy, U, reward, discount, row, col, actions, num_actions, threshold, whitespace_reward, k, plotter):
    iteration = 0
    while True:
        iteration+=1
        U = policyEvaluation(policy, U, reward, discount, row, col, actions, threshold, k)
        unchanged = True
        # iterate all states
        for r in range(row):
            for c in range(col):
                if U[r][c] == 99: # skip wall
                    policy[r][c] = 99 
                    continue
                # find action that maximise utility
                maxAction, maxU = None, -float("inf")
                for action in range(num_actions):
                    u = calculateU(U, r, c, reward, discount, row, col, actions, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculateU(U, r, c, reward, discount, row, col, actions, policy[r][c]):
                    policy[r][c] = maxAction # the action that maximizes the utility
                    unchanged = False
                plotter[f'{r}-{c}'].append(U[r][c])
        if unchanged: # if policy changed, stop policy iteration
            break
        printEnvironment(U, row, col, reward, whitespace_reward)
        printEnvironment(policy, row, col, reward, whitespace_reward)
    print(f'Total iteration {iteration}')
    return policy, iteration
#### End of Solution: Policy Iteration

#### main method
def main():
    # init variables
    num_actions = 4
    actions = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
    reward = None
    U = []
    plotter = {}

    # Get arguments
    parser = argparse.ArgumentParser(description='Markov Decision Process, a grid world example involving value and policy iteration.')
    parser.add_argument('--discount', type=float, default=0.99, help='discount for bellman equation.')
    parser.add_argument('--threshold', type=float, default=62, help='Max change for epsilon value.')
    parser.add_argument('--whitespace_reward', type=float, default=-0.04, help='Rewards for the white tiles.')
    parser.add_argument('--k', type=float, default=4, help='Number of iterations for policy evaluation.')
    parser.add_argument('--map', type=str, default='./map', help='Filepath to map. Must be in JSON format.')
    parser.add_argument('--algorithm', type=int, default=0, help='Choose \n 0: for value iteration \n 1: policy iteration')
    
    args = parser.parse_args()
    discount = args.discount
    threshold = args.threshold
    whitespace_reward = args.whitespace_reward
    # row = args.row
    # col = args.col
    map = args.map
    algorithm = args.algorithm
    k = args.k

    # Print the initial environment
    # Load map and reward
    with open(f'{map}.json', 'r') as f:
        U = json.load(f)
        if len(U) > 0:
            row = len(U)
            col = len(U[0])
            reward = [[0 for r in U[0]] for c in U]
            print(U)
            for r in range(row):
                for c in range(col):
                    if U[r][c] != 99: plotter[f'{r}-{c}'] = []
        else:
            print('Invalid map!')
        # reward = [[0 for i in range(len(U)+1)]for j in range(len(U)+1)]
    print("The initial U is:\n")
    printEnvironment(U, row, col, reward, whitespace_reward, init=True)

    if algorithm == 0:
        print("Value iteration selected \n")
        # Value iteration
        U, iteration = valueIteration(U, row, col, num_actions, threshold, discount, reward, actions, whitespace_reward, plotter)

        # Get the optimal policy
        policy = getOptimalPolicy(U, row, col, num_actions, reward, discount, actions)
    elif algorithm == 1:
        print("Policy iteration selected \n")
        policy = [[random.randint(0, 3) for j in range(col)] for i in range(row)] # random initialized policy

        # Policy iteration
        policy, iteration = policyIteration(policy, U, reward, discount, row, col, actions, num_actions, threshold, whitespace_reward, k, plotter)

    # Print the optimal policy
    print("The optimal policy is:\n")
    printEnvironment(policy, row, col, reward, whitespace_reward, True)

    # plot results on graph
    for r in range(row):
        for c in range(col):
            if U[r][c] != 99: 
                plotter[f'{r}-{c}'] = [x / (min(plotter[f'{r}-{c}']) + (max(plotter[f'{r}-{c}']) - min(plotter[f'{r}-{c}']))) for x in plotter[f'{r}-{c}']] # normalisation
                plt.plot([i for i in range(iteration)], plotter[f'{r}-{c}'])        
    plt.xlabel('Iterations')
    plt.ylabel('State Values')
    plt.show()

# Program entry point
if __name__ == "__main__":
    main()