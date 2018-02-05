#!/usr/bin/env python

#hopper.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#4 February 2018
#
#this ...  
#to execute:    ./hopper.py

#imports
import numpy as np
import random
import copy

#initialize the environment = dict containing all constants that describe the system
def initialize_environment(rn_seed, cat_strategy, max_moves):
    random.seed(rn_seed)
    environment = {'rn_seed':rn_seed, 'cat_strategy':cat_strategy, 'max_moves':max_moves}
    return environment

#initialize state with bug at origin and cat randomly placed
def initialize_state(environment):
    bug_xy = np.array([0.0, 0.0])
    stdev = 1.0
    x = random.normalvariate(0.0, stdev)
    y = random.normalvariate(0.0, stdev)
    cat_xy = np.array([x, y])
    state = {'cat_xy':cat_xy, 'bug_xy':bug_xy}
    return state

#calculate the bug-cat separation
def get_separation(state):
    bug_xy = state['bug_xy']
    cat_xy = state['cat_xy']
    delta = bug_xy - cat_xy
    separation = np.sqrt((delta**2).sum())
    return separation

#move the bug
def move_bug(environment, state):
    #random component of bug's movement
    stdev = 1.0
    random_distance = random.normalvariate(0.0, stdev)
    pi = np.pi
    random_angle = random.uniform(-pi, pi)
    bug_dx = random_distance*np.cos(random_angle)
    bug_dy = random_distance*np.sin(random_angle)
    #systematic component is away from cat
    bug_xy = state['bug_xy']
    cat_xy = state['cat_xy']
    delta = -(cat_xy - bug_xy)
    delta_x = delta[0]
    delta_y = delta[1]
    separation = np.sqrt(delta_x**2 + delta_y**2)
    systematic_distance = 1.0/np.sqrt((1/3.33)**2 + separation**2)
    systematic_angle = np.arctan2(delta_y, delta_x)
    bug_dx += systematic_distance*np.cos(systematic_angle)
    bug_dy += systematic_distance*np.sin(systematic_angle)
    bug_delta_xy = np.array([bug_dx, bug_dy])
    return bug_delta_xy

#move cat
def move_cat(environment, state):
    cat_strategy = environment['cat_strategy']
    bug_xy = state['bug_xy'].copy()
    cat_xy = state['cat_xy'].copy()
    if (cat_strategy == 'simple'):
        #cat leaps right at bug
        cat_delta_xy = bug_xy - cat_xy
    return cat_delta_xy

#move bug and cat
def update_state(environment, state):
    bug_delta_xy = move_bug(environment, state)
    cat_delta_xy = move_cat(environment, state)
    next_state = copy.deepcopy(state)
    next_state['bug_xy'] += bug_delta_xy
    next_state['cat_xy'] += cat_delta_xy
    return next_state, cat_delta_xy

#calculate reward = 1/(bug-cat separation)
def get_reward(state):
    separation = get_separation(state)
    reward = 1.0/separation
    return reward

#check game state = running, or too many moves
def get_game_state(N_moves, environment):
    game_state = 'running'
    max_moves = environment['max_moves']
    if (N_moves > max_moves):
        game_state = 'max_moves'
    return game_state

#build neural network
def build_model(N_inputs, N_neurons, N_outputs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Dense(N_neurons, input_shape=(N_inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(N_neurons))
    model.add(Activation('relu'))
    model.add(Dense(N_outputs))
    model.add(Activation('linear'))
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

