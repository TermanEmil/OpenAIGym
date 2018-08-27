import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt


def inverse_lerp(min_val, max_val, value):
    return (value - min_val) / (max_val - min_val)


def lerp(min_val, max_val, scalar):
    return (1.0 - scalar) * min_val + scalar * max_val;


def normalize_state(state, env):
    return [
        inverse_lerp(
            env.observation_space.low[0],
            env.observation_space.high[0],
            state[0]),
        inverse_lerp(
            -env.unwrapped.max_speed,
            env.unwrapped.max_speed,
            state[1])
    ]


def main():
    # Learning parameters.
    learning_rate = 0.00025
    q_discount = 0.99
    random_action_chance = 0.1
    num_episodes = 2000
    max_steps_per_episode = 1000000

    env = gym.make('MountainCarContinuous-v0')

    # Position and velocity inputs
    inputs_count = 2

    # 0 for left, 1 for right
    actions_count = 1

    tf.reset_default_graph()

    # The feed forward part of the network.
    inputs = tf.placeholder(shape=[1, inputs_count], dtype=tf.float32)

    weight_init_bounds = (-1, 1)
    weights = tf.Variable(
        tf.random_uniform(
            [inputs_count, actions_count],
            weight_init_bounds[0],
            weight_init_bounds[1]))

    q_out = tf.matmul(inputs, weights)
    prediction = tf.argmax(q_out, axis=1)

    # Init tf loss function. Loss = square difference between the target
    # and predicted Q values.
    next_q = tf.placeholder(shape=[1, actions_count], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_q - q_out))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update_model = trainer.minimize(loss=loss)

    # Init plot containers
    rewards_buf = []
    steps_per_episode_buf = []

    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        for episode_i in range(num_episodes):
            # Reset environment and get initial observation.
            s = env.reset()

            episode_reward = 0
            for step_i in range(max_steps_per_episode):
                # Choose greedily an action from the Q-network.
                a, all_q = session.run(
                    [prediction, q_out],
                    feed_dict={inputs: [normalize_state(s, env)]}
                )
                a = a.astype(float)

                # Chance to make a random action.
                # if np.random.rand(1) < random_action_chance:
                #     a[0] = np.random.rand(1)

                # a = np.array(a)
                # Get new state and reward from environment.
                final_force = lerp(1, -1, a[0])
                print(a[0], final_force)
                s1, r, done, _ = env.step([final_force])

                # Obtain the Q' values by feeding the new state
                # through the network.
                q1 = session.run(
                    q_out,
                    feed_dict={inputs: [normalize_state(s1, env)]}
                )

                target_q = all_q
                target_q[0] = r + q_discount * np.max(q1)

                # Train the network using target and predicted Q values.
                _, new_weights = session.run(
                    [update_model, weights],
                    feed_dict={
                        inputs: [normalize_state(s, env)],
                        next_q: target_q
                    })

                episode_reward += r
                s = s1
                if done:
                    # Reduce the chance of random action as the model is trained
                    random_action_chance = 1.0 / ((episode_i / 50) + 10)

                    steps_per_episode_buf.append(step_i)
                    break

                env.render()

            rewards_buf.append(episode_reward)

        # render_results(
        #     env,
        #     session,
        #     prediction,
        #     inputs,
        #     inputs_count,
        #     q_out,
        #     max_steps_per_episode)

    print(
        "Percent of successful episodes: ",
        sum(rewards_buf) / num_episodes,
        "%")

    plt.plot(rewards_buf)
    # plt.show()


if __name__ == '__main__':
    main()
