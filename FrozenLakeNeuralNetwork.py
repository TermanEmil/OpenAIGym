import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt


def render_results(
        env,
        session,
        prediction,
        inputs,
        states_count,
        q_out,
        max_episode_steps=100):

    s = env.reset()
    env.render()

    for i in range(max_episode_steps):
        a, _ = session.run(
            [prediction, q_out],
            feed_dict={inputs: create_binary_inputs(states_count, s)}
        )

        s1, _, done, _ = env.step(a[0])
        s = s1

        env.render()

        if done:
            break


# Create a vector of zeros with only one cell set to 1.
def create_binary_inputs(states_count, index_of_active_cell):
    return [np.identity(states_count)[index_of_active_cell]]


def main():
    # Learning parameters.
    learning_rate = 0.1
    q_discount = 0.95
    random_action_chance = 0.1
    num_episodes = 2000
    max_steps_per_episode = 100

    env = gym.make('FrozenLake-v0')

    # Value aliases.
    states_count = env.observation_space.n
    actions_count = env.action_space.n

    tf.reset_default_graph()

    # The feed forward part of the network.
    inputs = tf.placeholder(shape=[1, states_count], dtype=tf.float32)

    weight_init_bounds = (0, 0.01)
    weights = tf.Variable(
        tf.random_uniform(
            [states_count, actions_count],
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
                    feed_dict={inputs: create_binary_inputs(states_count, s)}
                )

                # Chance to make a random action.
                if np.random.rand(1) < random_action_chance:
                    a[0] = env.action_space.sample()

                # Get new state and reward from environment.
                s1, r, done, _ = env.step(a[0])

                # Obtain the Q' values by feeding the new state
                # through the network.
                q1 = session.run(
                    q_out,
                    feed_dict={inputs: create_binary_inputs(states_count, s1)}
                )

                target_q = all_q
                target_q[0, a[0]] = r + q_discount * np.max(q1)

                # Train the network using target and predicted Q values.
                _, new_weights = session.run(
                    [update_model, weights],
                    feed_dict={
                        inputs: create_binary_inputs(states_count, s),
                        next_q: target_q
                    })

                episode_reward += r
                s = s1
                if done:
                    # Reduce the chance of random action as the model is trained
                    random_action_chance = 1.0 / ((episode_i / 50) + 10)

                    steps_per_episode_buf.append(step_i)
                    break

            rewards_buf.append(episode_reward)

        render_results(
            env,
            session,
            prediction,
            inputs,
            states_count,
            q_out,
            max_steps_per_episode)

    print(
        "Percent of successful episodes: ",
        sum(rewards_buf) / num_episodes,
        "%")

    plt.plot(rewards_buf)
    plt.show()


if __name__ == '__main__':
    main()
