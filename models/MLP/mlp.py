from __future__ import absolute_import, division, print_function, unicode_literals
import os # file paths, command line arguments, enviroment variables
import tensorflow as tf # linear algebra with tensors for machine learning with CPU and GPU
import cProfile # for profiling in long-running programs
import numpy as np

# uncomment this line for eager execution
# tensors are evaluated when called without a session
# tf.enable_eager_execution()

def layered_mlp(learning_rate = 0.01):
    # session scope
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    # declare input and desired output
    X = tf.placeholder(tf.float32)

    desired_output = tf.placeholder(tf.float32)

    # network variables to be trained [number of outputs, number of inputs], 
    W1 = tf.Variable(tf.random_uniform([2,2], minval=0, maxval=1, dtype=tf.float32), trainable=True)
    W2 = tf.Variable(tf.random_uniform([3,2], minval=0, maxval=1, dtype=tf.float32), trainable=True)
    W3 = tf.Variable(tf.random_uniform([2,3], minval=0, maxval=1, dtype=tf.float32), trainable=True)

    # bias for each layer
    B1 = tf.Variable(tf.random_uniform([2,1], minval=0, maxval=1, dtype=tf.float32), trainable=False)
    B2 = tf.Variable(tf.random_uniform([3,1], minval=0, maxval=1, dtype=tf.float32), trainable=False)
    B3 = tf.Variable(tf.random_uniform([2,1], minval=0, maxval=1, dtype=tf.float32), trainable=False)

    # created model
    # No. of cols from A should be = No. of lines from B to properly 
    # perform matrix multiplication and obtain the correct output vector shape
    W1_X = tf.matmul(W1, X, name="W1_X") + B1
    A1 = tf.nn.relu(W1_X, name="Activation_W1")
    W2_A1 = tf.matmul(W2, A1, name="W2_A1") + B2
    A2 = tf.nn.relu(W2_A1, name="Activation_W2")
    W3_A2 = tf.matmul(W3, A2, name="W3_A2") + B3
    A3 = tf.nn.relu(W3_A2, name="Activation_W3")

    # the prediction for this model is the sum of the nodes for the last layer
    prediction = tf.reduce_sum(A3)

    # uncomment this line if you want a softmax for the last activation layer
    # S1 = tf.nn.softmax(A3, name="Softmax_W1")

    # error/loss to be optimized
    loss = tf.square(desired_output - prediction)

    # gradient calculator and optimizer
    # it will take care of the calculation and to apply the derivatives
    # of all the trainable variables in the network model
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # computing gradients given a loss and a list of variables - it is a list containing tuples
    gradients = opt.compute_gradients(loss, [W1, W2, W3])

    # apply the calculated gradients for each trainable variable in the network
    train = opt.apply_gradients(gradients)

    # this is an automatic gradient descent method
    # uncomment this line if you want to test it
    #train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # global variables initialization
    init = tf.global_variables_initializer()

    # defining training dataset
    dataset = np.array([[.1, .9, .1],
            [.2, .8, .2],
            [.3, .7, .3],
            [.4, .6, .4],
            [.5, .5, .5],
            [.6, .4, .6],
            [.7, .3, .7],
            [.8, .2, .8],
            [.9, .1, .9]])

    # manually train the network
    with tf.Session() as session:
        # firstly you need to initialize the network basic variables
        session.run(init)

        # loop the number of epochs 
        # 1 epoch is an iteration over all the training data
        for i in range(1000):
            print("\n------------------------ Epoch: " + str(i) + "------------------------\n")
            data_index = 0
            for data_row in dataset:
                # just getting the right rows to input into the network
                input_X = np.reshape(data_row[:2], (2,1))

                # getting the right rows for correct prediction for training
                desired_prediction = data_row[2:3]

                # run the created graph containing all tensors
                # that define the model
                out_loss, _ = session.run(
                    # this list contains all tensors to run explicitly
                    [loss,
                    train,
                    # print within a tensor node example bellow
                    # uncomment this line if you want to see what it does
                    # tf.Print(loss, [loss], "tensor print example - loss: ")
                    ], 
                    # the feed_dict is you feed the model's placeholders
                    # they are basically the input variables
                    feed_dict={X: input_X, desired_output: desired_prediction})

                # print the results of an execution with several parameters
                print("loss: " + str(out_loss[0]))

                # increment with data is being processed
                data_index = data_index + 1

    # session end
    session.close()

if __name__ == '__main__':
    # run with cProfile for long-running tensorflow graph creationg and training operations
    cProfile.run('layered_mlp()', 'mlp_profile.out')