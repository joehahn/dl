#!/usr/bin/env python

#gridworld_0jmh.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#31 January 2018
#
#this was adapted from http://outlace.com/rlpart3.html  
#to execute:    ./gridworld_0jmh.py

#initialize state = dict containing x,y coordinates of all objects in the system
def initialize_state():
    state = {'agent':{'x':3, 'y':1}, 'goal':{'x':1, 'y':0}, 'pit':{'x':1, 'y':3}, 'wall':{'x':2, 'y':2}}
    return state

#initialize the environment = dict containing all other constants that describe the system
def initialize_environment(state, grid_size):
    actions = ['up', 'down', 'left', 'right']
    action_indexes = range(len(actions))
    objects = state.keys()
    environment = {'actions':actions, 'action_indexes':action_indexes, 'objects':objects,
        'grid_size':grid_size}
    return environment

#define agent's possible actions
def move_agent(state, action, environment):
    agent = state['agent'].copy()
    next_state = state.copy()
    grid_size = environment['grid_size']
    if (action == 'up'):
        if (agent['y'] < grid_size-1):
            agent['y'] += 1
    if (action == 'down'):
        if (agent['y'] > 0):
            agent['y'] -= 1
    if (action == 'right'):
        if (agent['x'] < grid_size-1):
            agent['x'] += 1
    if (action == 'left'):
        if (agent['x'] > 0):
            agent['x'] -= 1
    wall = state['wall']
    if (agent != wall):
        next_state['agent'] = agent
    return next_state

#get reward...+10 if agent is at goal, -10 if agent is in pit, -1 otherwise
def get_reward(state):
    if (state['agent'] == state['goal']):
        return 10
    if (state['agent'] == state['pit']):
        return -10
    return -1

#generate 2D string array showing locations of all objects
import numpy as np
def state_grid(state, environment):
    grid_size = environment['grid_size']
    grid = np.zeros((grid_size, grid_size), dtype='string')
    objects = environment['objects']
    for object in objects:
        xy = state[object]
        x = xy['x']
        y = xy['y']
        grid[y, x] = object[0].upper()
    return grid

#convert state into a numpy array of x,y coordinates
def state2vector(state, environment):
    vector = []
    for object in objects:
        xy = state[object]
        x = xy['x']
        y = xy['y']
        vector += [x, y]
    return np.array(vector)

#initial conditions
grid_size = 7
state = initialize_state()
environment = initialize_environment(state, grid_size)
objects = environment['objects']
actions = environment['actions']
action_indexes = environment['action_indexes']
state_vector = state2vector(state, environment)
print 'objects = ', objects
print 'actions = ', actions
print 'action_indexes = ', action_indexes
print 'state = ', state
print 'state_vector = ', state_vector
print state_grid(state, environment)
N_inputs = len(state_vector)
N_outputs = len(actions)
grid_size = environment['grid_size']
print 'N_inputs = ', N_inputs
print 'N_outputs = ', N_outputs
print 'grid_size = ', grid_size

#build neural network
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(N_inputs,)))
model.add(Activation('relu'))
model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(N_outputs, kernel_initializer='lecun_uniform'))
model.add(Activation('linear'))
rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
print model.summary()

#train
N_training_games = 1000
gamma = 0.9
epsilon = 1.0
for N_game in range(N_training_games):
    state = initialize_state()
    N_turns = 0
    if (N_game > 10):
        if (epsilon > 0.1):
            epsilon -= 1.0/N_game
    while (True):
        state_vector = state2vector(state, environment)
        state_vector_reshaped = state_vector.reshape(1, N_inputs)
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state_vector_reshaped, batch_size=1)
        if (np.random.random() < epsilon):
            #choose a random action_index
            action_index = np.random.choice(action_indexes)
        else:
            #choose best action_index from Q(s,a) values
            action_index = np.argmax(qval)
        action = actions[action_index]
        new_state = move_agent(state, action, environment)
        reward = get_reward(new_state)
        #Get max_Q(S',a)
        new_state_vector = state2vector(new_state, environment)
        new_state_vector_reshaped = new_state_vector.reshape(1, N_inputs)
        newQ = model.predict(new_state_vector_reshaped, batch_size=1)
        maxQ = np.max(newQ)
        if (reward == -1):
            #non-terminal state
            update = reward + gamma*maxQ
        else:
            #this is a terminal state
            update = reward
        qval[0][action_index] = update
        print("Game number: %s" % N_game)
        print("turn number: %s" % N_turns)
        print("epsilon: %s" % epsilon)
        print("reward: %s" % reward)
        print("action: %s" % action)
        model.fit(new_state_vector_reshaped, qval, batch_size=1, epochs=1, verbose=0)
        state = new_state
        N_turns += 1
        if (reward != -1):
            print("game completed")
            break
        if (N_turns > 3*grid_size):
            print("too many turns")
            break

#test
def test_model(grid_size):
    state = initialize_state()
    print('initial state:')
    print state_grid(state, environment)
    print ("=======================")
    N_turns = 0
    while (True):
        state_vector = state2vector(state, environment)
        state_vector_shaped = state_vector.reshape(1, N_inputs)
        qval = model.predict(state_vector_shaped, batch_size=1)
        action_index = np.argmax(qval)
        action = actions[action_index]
        state = move_agent(state, action, environment)
        N_turns += 1
        print("turn: %s    action: %s" %(N_turns, action))
        print state_grid(state, environment)
        reward = get_reward(state)
        if reward != -1:
            print("reward: %s" % reward)
            break
        if (N_turns > 9):
            print("game lost, too many actions")
            break

test_model(grid_size)