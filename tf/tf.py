import tensorflow as tf
x_init = [1.0, 2.0]
x = tf.Variable(x_init, name='x', dtype=tf.float32)
y_vals = [0.5, 3.0]
y = tf.Variable(y_vals, name='y', dtype=tf.float32)
d = tf.abs(x - y)
s = tf.reduce_sum(d)
optimizer = tf.train.GradientDescentOptimizer(0.25)
train = optimizer.minimize(s)
init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init)    
    print("starting at", "x:", session.run(x), "s(x):", session.run(s))
    for step in range(10):  
        session.run(train)
        print("step", step, "x:", session.run(x), "s(x):", session.run(s))

import tensorflow as tf
import numpy as np
def y():
    p0 = np.array([1.0, 2.0])
    sigma = 0.1*p0
    return np.random.normal(loc=p0, scale=sigma)

x_init = [2.0, 1.0]
x = tf.Variable(x_init, name='x', dtype=tf.float32)
d = tf.abs(x - y())
s = tf.reduce_sum(d)
optimizer = tf.train.GradientDescentOptimizer(0.25)
train = optimizer.minimize(s)
init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init)    
    print("starting at", "x:", session.run(x), "s(x):", session.run(s))
    for step in range(20):  
        session.run(train)
        print("step", step, "x:", session.run(x), "s(x):", session.run(s))

#from https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
import numpy as np
from matplotlib import pyplot as plt
import seaborn
# Define input data
X_data = np.arange(100, step=0.1)
y_data = X_data + 20*np.sin(X_data/10)
# Plot input data
plt.scatter(X_data, y_data)
plt.show()

# Define data size and batch size
import tensorflow as tf
n_samples = 1000 
batch_size = 100 
# Tensorflow is finicky about shapes, so resize 
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))
# Define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))
# Define variables to be learned
with tf.variable_scope( "linear-regression" ):
    W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable( "bias", (1, ), initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pred)**2/n_samples)

# Sample code to run one step of gradient descent
opt_operation = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    # Initialize Variables in graph
    sess.run(tf.initialize_all_variables())
    # Gradient descent loop for 500 steps
    for _ in range(500):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices] 
        # Do gradient descent step
        yp, loss_val = sess.run([opt_operation, loss], feed_dict={X:X_batch, y:y_batch})

# Plot input data
plt.scatter(X_data, y_data)
plt.show()


#from https://www.activestate.com/blog/2017/10/using-tensorflow-predictive-analytics-linear-regression
#and https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
import numpy as np
from matplotlib import pyplot as plt
import seaborn
# Define input data
X_data = np.arange(100, step=0.1)
y_data = X_data + 20*np.sin(X_data/10)
# Plot input data
plt.scatter(X_data, y_data)
plt.show()
import tensorflow as tf
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
Y_pred = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)
n_epochs = 1000
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.initialize_all_variables())

with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.initialize_all_variables())
    # Fit all training data
    prev_training_loss = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_loss = sess.run(
            loss, feed_dict={X: xs, Y: ys})
        print(training_loss)
        if epoch_i % 20 == 0:
            ax.plot(xs, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                    'k', alpha=epoch_i / n_epochs)
            fig.show()
            plt.draw()
        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_loss - training_loss) < 0.000001:
            break
        prev_training_loss = training_loss


#linear regression example from 
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
# Training Data
train_X = np.arange(100, step=0.1)
train_Y = X_data + 20*np.sin(X_data/10)
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")
# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


#generate location and reward data
import numpy as np
N_agents = 2
N_locations = 3
N_samples = 1000
p0 = np.arange(N_locations, dtype=float) + 1.0
sigma = 0.5*p0
locations_list = []
rewards_list = []
for idx in range(N_samples):
    locations = np.random.randint(0, N_locations, size=N_agents)
    l = p0[locations]
    s = sigma[locations]
    reward = np.random.normal(loc=l, scale=s).sum()
    locations_vector = np.zeros(N_locations, dtype=float)
    for loc in locations:
        locations_vector[loc] += 1.0
    locations_list += [locations_vector]
    rewards_list += [reward]
locations = np.array(locations_list)
rewards = np.array(rewards_list)

#test-train-validation split
rn_seed = 13
train_fraction = 0.67
from sklearn.model_selection import train_test_split
x_train, x_test_validate, y_train, y_test_validate = \
    train_test_split(locations, rewards, train_size=train_fraction, random_state=rn_seed)
x_test, x_validate, y_test, y_validate = \
    train_test_split(x_test_validate, y_test_validate, train_size=train_fraction, random_state=rn_seed)
print locations.shape
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape
print x_validate.shape, y_validate.shape

#this helper function builds an MLP neural network
def mlp_model(layers):
    from keras.models import Sequential
    model = Sequential()
    from keras.layers import Dense
    N = layers[0]
    model.add(Dense(N, activation='elu', input_shape=(N,)))
    for N in layers[1:-1]:
        model.add(Dense(N, activation='elu'))
    N = layers[-1]
    model.add(Dense(N, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#make MLP model
N_inputs = N_locations
N_outputs = 1
layers = [N_inputs, N_agents*N_locations, N_agents*N_locations/2, N_outputs]
print 'layers = ', layers
model = mlp_model(layers)
model.summary()

N_epochs = 100
batch_size = N_samples/10
fit_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=N_epochs, verbose=1, 
    validation_data=(x_validate, y_validate))
