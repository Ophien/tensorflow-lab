'''
MIT License

Copyright (c) 2020 Alysson Ribeiro da Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os # file paths, command line arguments, enviroment variables, system calls
import tensorflow as tf # linear algebra with tensors for machine learning with CPU and GPU
import cProfile # for profiling in long-running programs
import numpy as np # to use numpy arrays and general operations
import matplotlib.pyplot as plt # to plot charts with errors and related stuff

# uncomment this line for eager execution
# tensors are evaluated when called without a session
# tf.enable_eager_execution()

def build_network_model():
    # declare input and desired output
    X = tf.placeholder(tf.float32)

    # define all weights
    weights = []
    biases = []

    # network variables to be trained [number of outputs, number of inputs], 
    weights.append(tf.Variable(tf.random_uniform([2,2], minval=0, maxval=1, dtype=tf.float32), trainable=True, name="W1"))
    weights.append(tf.Variable(tf.random_uniform([3,2], minval=0, maxval=1, dtype=tf.float32), trainable=True, name="W2"))
    weights.append(tf.Variable(tf.random_uniform([2,3], minval=0, maxval=1, dtype=tf.float32), trainable=True, name="W3"))

    # bias for each layer
    biases.append(tf.Variable(tf.random_uniform([2,1], minval=0, maxval=1, dtype=tf.float32), trainable=False))
    biases.append(tf.Variable(tf.random_uniform([3,1], minval=0, maxval=1, dtype=tf.float32), trainable=False))
    biases.append(tf.Variable(tf.random_uniform([2,1], minval=0, maxval=1, dtype=tf.float32), trainable=False))

    # created model
    # No. of cols from A should be = No. of lines from B to properly 
    # perform matrix multiplication and obtain the correct output vector shape
    W1_X = tf.matmul(weights[0], X, name="W1_X") + biases[0]
    A1 = tf.nn.relu(W1_X, name="Activation_W1")
    W2_A1 = tf.matmul(weights[1], A1, name="W2_A1") + biases[1]
    A2 = tf.nn.relu(W2_A1, name="Activation_W2")
    W3_A2 = tf.matmul(weights[2], A2, name="W3_A2") + biases[2]
    A3 = tf.nn.relu(W3_A2, name="Activation_W3")

    # the prediction for this model is the sum of the nodes for the last layer
    prediction = tf.reduce_sum(A3)

    return X, weights, biases, prediction

def build_training_model(model_graph_for_prediction, tensors_with_weights, learning_rate = 0.01):
    desired_output = tf.placeholder(tf.float32)

    # uncomment this line if you want a softmax for the last activation layer
    # S1 = tf.nn.softmax(A3, name="Softmax_W1")

    # error/loss to be optimized
    loss = tf.square(desired_output - model_graph_for_prediction)

    # gradient calculator and optimizer
    # it will take care of the calculation and to apply the derivatives
    # of all the trainable variables in the network model
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    # computing gradients given a loss and a list of variables - it is a list containing tuples
    gradients = opt.compute_gradients(loss, 
        [tensors_with_weights[0], 
         tensors_with_weights[1], 
         tensors_with_weights[2]])

    # apply the calculated gradients for each trainable variable in the network
    train = opt.apply_gradients(gradients)

    # this is an automatic gradient descent method
    # uncomment this line if you want to test it
    #train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return loss, train, desired_output

def train(learning_rate = 0.01):

    #----------------------------------------------------------------------------------------
    #----------------------------- Defining or loading all data -----------------------------
    #----------------------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------------------
    #----------------------------------- Defining models ------------------------------------
    #----------------------------------------------------------------------------------------

    # average loss list
    global_average_loss = []

    # model tensors, all tensors are needed to feed the train model correctly 
    # using the run function
    X, weights, _, prediction_model = build_network_model()
    loss, training_model, desired_output = build_training_model(prediction_model, weights, learning_rate=learning_rate)

    #----------------------------------------------------------------------------------------
    #---------------------------- Saver object to save trained model ------------------------
    #----------------------------------------------------------------------------------------

    # saver object used to save and load the model graph and variables
    saver = tf.train.Saver()

    #----------------------------------------------------------------------------------------
    #--------------------------------------- Saving model -----------------------------------
    #----------------------------------------------------------------------------------------

    # uncomment this line to use sessions explicitly
    #session = tf.Session()

    # creating session
    with tf.Session() as session:

        # initialization should be done after creating all variables
        session.run(tf.initialize_all_variables())

        # loop the number of epochs 
        # 1 epoch is an iteration over all the training data
        step = 0
        for i in range(1000):
                
            data_index = 0
            cumulative_loss = []

            for data_row in dataset:
                # just getting the right rows to input into the network
                input_X = np.reshape(data_row[:2], (2,1))

                # getting the right rows for correct prediction for training
                desired_prediction = data_row[2:3]

                #----------------------------------------------------------------------------------------
                #-------------- Running model to train the network with the input data ------------------
                #----------------------------------------------------------------------------------------

                out_loss, _, _ = session.run(
                    # this list contains all tensors to run explicitly
                    [
                    loss,
                    prediction_model,
                    training_model,
                    # print within a tensor node example bellow
                    # uncomment this line if you want to see what it does
                    # tf.Print(loss, [loss], "tensor print example - loss: ")
                    ], 
                    # the feed_dict is you feed the model's placeholders
                    # they are basically the input variables
                    feed_dict={X: input_X, desired_output: desired_prediction})

                # append sample loss to the cumulative loss list
                cumulative_loss.append(out_loss[0])

                # increment with data is being processed
                data_index = data_index + 1

            # calculates and prints the loss average
            epoch_average_loss = np.average(cumulative_loss)
            global_average_loss.append(epoch_average_loss)

            # increment step for saver
            step = step + 1

            print("Epoch: " + str(i) + " Average loss: " + str(epoch_average_loss))

        #----------------------------------------------------------------------------------------
        #--------------------------------------- Saving model -----------------------------------
        #----------------------------------------------------------------------------------------

        # plot loss behavior
        plt.plot([i for i in global_average_loss])
        plt.savefig("./generated_charts/average_errors.png")

        # save trained model
        # create tree files
        # data: all variable values
        # meta: the model graph
        # index: the checkpoints
        saver.save(session, "./trained_model/mlp_saved_model", global_step=step)

        # save tensorboard logs for debug
        tf.summary.FileWriter('./tensorboard_logs/graphs', session.graph)

    # uncomment this line if you are using sessions explicitly
    #session.close()

if __name__ == '__main__':
    # run with cProfile for long-running tensorflow graph creationg and training operations
    #cProfile.run('train()', './cprofile_logs/mlp_profile.out')
    train()