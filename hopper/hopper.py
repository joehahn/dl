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
def initialize_environment(rn_seed, max_moves):
    random.seed(rn_seed)
    environment = {'rn_seed':rn_seed, 'max_moves':max_moves}
    return environment

def initialize_state(environment):
    bug_xy = np.array([0.0, 0.0])
    stdev = 1.0
    x = random.normalvariate(0.0, stdev)
    y = random.normalvariate(0.0, stdev)
    cat_xy = np.array([x, y])
    state = {'cat_xy':cat_xy, 'bug_xy':bug_xy}
    return state

def move_bug(bug_xy, cat_xy):
    #random component of bug's movement
    stdev = 1.0
    distance = random.normalvariate(0.0, stdev)
    pi = np.pi
    direction_angle = random.uniform(-pi, pi)
    bug_dx = distance*np.cos(direction_angle)
    bug_dy = distance*np.sin(direction_angle)
    #system component is away from cat
    delta_x = cat_xy[0] - bug_xy[0]
    delta_y = cat_xy[0] - bug_xy[0]
    delta = np.sqrt(delta_x**2 + delta_y**2)
    distance = 1.0/delta
    distance_max = 3.33
    if (distance > distance_max):
        distance = distance_max
    direction_angle = np.arctan2(delta_y, delta_x)
    bug_dx += distance*np.cos(direction_angle)
    bug_dy += distance*np.sin(direction_angle)
    bug_delta_xy = np.array([bug_dx, bug_dy])
    return bug_delta_xy

def update_state(state, cat_delta_xy):
    bug_xy = state['bug_xy'].copy()
    cat_xy = state['cat_xy'].copy()
    bug_delta_xy = move_bug(bug_xy, cat_xy)
    next_state = copy.deepcopy(state)
    next_state['bug_xy'] += bug_delta_xy
    next_state['cat_xy'] += cat_delta_xy
    return next_state

#initial settings
rn_seed = 12
max_moves = 10

#initialize system
environment = initialize_environment(rn_seed, max_moves)
state = initialize_state(environment)
print 'environment = ', environment
print 'state = ', state

for move in range(max_moves):
    cat_xy = state['cat_xy'].copy()
    bug_xy = state['bug_xy'].copy()
    cat_delta_xy = bug_xy - cat_xy
    next_state = update_state(state, cat_delta_xy)
    print next_state
    state = copy.deepcopy(next_state)


