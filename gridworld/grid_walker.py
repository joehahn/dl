#!/usr/bin/env python

#gridwalker.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#31 January 2018
#
#this was adapted from http://outlace.com/rlpart3.html  
#to execute:    ./gridwalker.py

#imports
import numpy as np
import random

#initialize the environment = dict containing all constants that describe the system
def initialize_environment(grid_size):
    actions = ['up', 'down', 'left', 'right']
    action_indexes = range(len(actions))
    objects = ['agent', 'goal', 'pit', 'wall']
    max_moves = 4*grid_size
    environment = {'actions':actions, 'action_indexes':action_indexes, 'objects':objects,
        'grid_size':grid_size, 'max_moves':max_moves}
    return environment

#initialize state = dict containing x,y coordinates of all objects in the system,
#with agent's location random and all other objects in fixed location
def initialize_state(environment):
    wall = {'x':1, 'y':3}
    pit  = {'x':3, 'y':2}
    goal = {'x':4, 'y':4}
    grid_size = environment['grid_size']
    while (True):
        agent = {'x':np.random.randint(0, grid_size), 'y':np.random.randint(0, grid_size)}
        if (agent != wall):
            if (agent != pit):
                if (agent != goal):
                    break
    state = {'agent':agent, 'wall':wall, 'pit':pit, 'goal':goal}
    return state

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

#get reward
def get_reward(current_state, previous_state):
    if (current_state['agent'] == current_state['goal']):
        #agent is at goal
        return 10
    if (current_state['agent'] == current_state['pit']):
        #agent is in pit
        return -10
    if (current_state == previous_state):
        #agent was blocked by a wall or boundary
        return -3
    return -1

#check if game is still on or over
def check_game_on(state, N_moves, environment):
    agent = state['agent']
    goal = state['goal']
    pit = state['pit']
    max_moves = environment['max_moves']
    game_on = True
    if (agent == goal):
        game_on = False
    if (agent == pit):
        game_on = False
    if (N_moves > max_moves):
        game_on = False
    return game_on

#generate 2D string array showing locations of all objects
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
    return np.array(vector).reshape(1, len(vector))

#check initial conditions
grid_size = 6
rn_seed = 15
np.random.seed(rn_seed)
environment = initialize_environment(grid_size)
state = initialize_state(environment)
objects = environment['objects']
actions = environment['actions']
action_indexes = environment['action_indexes']
state_vector = state2vector(state, environment)
grid = state_grid(state, environment)
print 'objects = ', objects
print 'actions = ', actions
print 'action_indexes = ', action_indexes
print 'state = ', state
print 'state_vector = ', state_vector
print 'state_vector.shape = ', state_vector.shape
print np.rot90(grid.T)
N_inputs = state_vector.shape[1]
N_outputs = len(actions)
grid_size = environment['grid_size']
max_moves = environment['max_moves']
print 'N_inputs = ', N_inputs
print 'N_outputs = ', N_outputs
print 'grid_size = ', grid_size
print 'max_moves = ', max_moves
print 'rn_seed = ', rn_seed

##build neural network
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.advanced_activations import LeakyReLU, PReLU
#from keras.optimizers import RMSprop
#model = Sequential()
#model.add(Dense(16*N_inputs, input_shape=(N_inputs,)))
#model.add(LeakyReLU(alpha=0.01))
#model.add(Dense(12*N_inputs))
#model.add(LeakyReLU(alpha=0.01))
#model.add(Dense(N_outputs))
#model.add(Activation('linear'))
#rms = RMSprop()
#model.compile(loss='mse', optimizer=rms)
#print model.summary()

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

#initialize history with random moves
history_size = 1000
from collections import deque
history = deque(maxlen=history_size)
state = initialize_state(environment)
N_moves = 0
while (len(history) < history_size):
    state_vector = state2vector(state, environment)
    action_index = np.random.choice(action_indexes)
    action = actions[action_index]
    next_state = move_agent(state, action, environment)
    reward = get_reward(next_state, state)
    game_on = check_game_on(next_state, N_moves, environment)
    history.append((state, action, reward, next_state, game_on))
    if (game_on):
        state = next_state
        N_moves += 1
    else:
        state = initialize_state(environment)
        N_moves = 0

#train
N_training_games = 1000
gamma = 0.9
epsilon = 1.0
action_indexes = environment['action_indexes']
for N_games in range(N_training_games):
    state = initialize_state(environment)
    N_moves = 0
    if (N_games > 20):
        #slowly ramp epsilon down to 0.1
        if (epsilon > 0.1):
            epsilon -= 1.0/(N_training_games/2)
    game_on = check_game_on(state, N_moves, environment)
    while (game_on):
        #begin experience replay
        batch_size = history_size/10
        history_sub = random.sample(history, batch_size)
        statez = [h[0] for h in history_sub]
        actionz = [h[1] for h in history_sub]
        rewardz = [h[2] for h in history_sub]
        statez_next = [h[3] for h in history_sub]
        game_onz = [h[4] for h in history_sub]
        state_vectorz = np.array([state2vector(s, environment) for s in statez]).reshape(batch_size, N_inputs)
        Q = model.predict(state_vectorz, batch_size=batch_size)
        state_vectorz_next = np.array([state2vector(s, environment) for s in statez_next]).reshape(batch_size, N_inputs)
        Q_next = model.predict(state_vectorz_next, batch_size=batch_size)
        for idx in range(batch_size):
            reward = rewardz[idx]
            max_Q_next = np.max(Q_next[idx])
            action = actionz[idx]
            for j in action_indexes:
                if (actionz[j] == action):
                    Q[j] = reward
                    if (game_onz[idx]):
                        Q[j] += gamma*max_Q_next
        model.fit(state_vectorz, Q, batch_size=batch_size, epochs=1, verbose=0)
        #end experience replay
        state_vector = state2vector(state, environment)
        Q = model.predict(state_vector, batch_size=1)
        if (np.random.random() < epsilon):
            #choose a random action_index
            action_index = np.random.choice(action_indexes)
        else:
            #choose best action_index from Q(s,a) values
            action_index = np.argmax(Q)
        action = actions[action_index]
        state_next = move_agent(state, action, environment)
        state_vector_next = state2vector(state_next, environment)
        Q_next = model.predict(state_vector_next, batch_size=1)
        max_Q_next = np.max(Q_next)
        reward = get_reward(state_next, state)
        game_on = check_game_on(state_next, N_moves, environment)
        revised_Q = reward
        if (game_on):
            revised_Q += gamma*max_Q
        Q[0][action_index] = revised_Q
        if (game_on):
            pass
        else:
            print("game number: %s" % N_games)
            print("move number: %s" % N_moves)
            print("action: %s" % action)
            print("reward: %s" % reward)
            print("epsilon: %s" % epsilon)
            if (N_moves > environment['max_moves']):
                print("too many turns")
            else:
                print("game over")            
        history.append((state, action, reward, next_state, game_on))
        state = next_state
        N_moves += 1



#test
def test(environment):
    state = initialize_state(environment)
    grid = state_grid(state, environment)
    print('initial state:')
    print np.rot90(grid.T)
    print ("=======================")
    N_moves = 0
    game_over = False
    while (game_over == False):
        state_vector = state2vector(state, environment)
        Q = model.predict(state_vector, batch_size=1)
        action_index = np.argmax(Q)
        action = actions[action_index]
        next_state = move_agent(state, action, environment)
        N_moves += 1
        print("move: %s    action: %s" %(N_moves, action))
        grid = state_grid(next_state, environment)
        print np.rot90(grid.T)
        reward = get_reward(next_state, state)
        state = next_state
        print("reward: %s" % reward)
        if (reward > -5) and (reward < 5):
            #non-terminal state
            pass
        else:
            #terminal state
            game_over = True
            print("game completed")
        if (N_moves > environment['max_moves']):
            print("game lost, too many moves")
            game_over = True

test(environment)