import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Learning parameters.
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # Declare training data placeholders.
    # - Input - for 28x28 pixels
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])

    # - Output data - 10 digits.
    target_out = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # Declaring the weights.
    # - From input to hidden layer.
    layer1_neurons = 300
    layer1 = {
        "weights": tf.Variable(
            tf.random_normal(
                shape=[28 * 28, layer1_neurons],
                stddev=0.03),
            name='W1'),
        "bias": tf.Variable(tf.random_normal([layer1_neurons]), name='b1')
    }

    # - From hidden layer to output.
    out_layer = {
        "weights": tf.Variable(
            tf.random_normal(
                shape=[layer1_neurons, 10],
                stddev=0.03),
            name='W2'),
        "bias": tf.Variable(tf.random_normal([10]), name='b2')
    }

    # Calculate the output of the hidden layer
    hidden_out =\
        tf.nn.relu(
            tf.add(
                tf.matmul(inputs, layer1["weights"]),
                layer1["bias"]
            )
        )

    # Calculate the outputs.
    outputs =\
        tf.nn.softmax(
            tf.add(
                tf.matmul(hidden_out, out_layer["weights"]),
                out_layer["bias"]
            )
        )

    # Clip the output so that we don't have log(0).
    out_clipped = tf.clip_by_value(outputs, 1e-10, 0.999999)

    # Compute loss.
    error_sum = (
        target_out * tf.log(out_clipped) +
        (1 - target_out) * tf.log(1 - out_clipped)
    )
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(error_sum, axis=1))

    # Add an optimiser.
    optimiser = tf.train\
        .GradientDescentOptimizer(learning_rate)\
        .minimize(cross_entropy)

    # Define an accuracy assessment operation.
    correct_prediction = tf.equal(
        tf.argmax(target_out, axis=1),
        tf.argmax(outputs, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
    # Start the training.
    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                in_batch, out_batch = mnist.train.next_batch(batch_size)
                _, cost = sess.run(
                    [optimiser, cross_entropy],
                    feed_dict={inputs: in_batch, target_out: out_batch}
                )

                avg_cost += cost / total_batch
            print("Epoch: {}; cost = {:.3f}".format(epoch, avg_cost))

        print(
            sess.run(
                accuracy,
                feed_dict={
                    inputs: mnist.test.images,
                    target_out: mnist.test.labels}
            )
        )
