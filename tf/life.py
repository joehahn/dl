import tensorflow as tf
from matplotlib import pyplot as plt

shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    X = session.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
plt.show()

import numpy as np
from scipy.signal import convolve2d

def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X

with tf.Session() as session:
    initial_board_values = session.run(initial_board)
    X = session.run(board_update, feed_dict={board: initial_board_values})[0]

import matplotlib.animation as animation
def game_of_life(*args):
    X = session.run(board_update, feed_dict={board: X})[0]
    plot.set_array(X)
    return plot,

ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
plt.show()
