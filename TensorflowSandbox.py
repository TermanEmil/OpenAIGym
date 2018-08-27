import numpy as np
import tensorflow as tf


"""
A small and clear mini introduction to Tensorflow.
"""

if __name__ == '__main__':
    # Vars
    const = tf.constant(2.0, name='const')
    b = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='b')
    c = tf.Variable(1.0, name='c')

    # Operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    # Setup the variable initialization.
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
        print("a = ", a_out)

        a_out = sess.run(a, feed_dict={b: np.arange(0, 20)[:, np.newaxis]})
        print("a = ", a_out)

        e_out = sess.run(e)
        print("e = ", e_out)
