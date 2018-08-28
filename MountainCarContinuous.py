import gym
import numpy as np
import matplotlib.pyplot as plt


"""
I tried to solve this problem using Tabular Q-Learning. I divided the
observations into N positions and M velocities. The actions represent a table
of 2  values.

Observation:
    If the action table has 4++ values, the problem won't converge. I think
    that's because it need's a smooth transition from observations to actions.
    I believe a neural network will work much better. 
"""


# Mountain Car utils
def index_of(value, table):
    return min(range(len(table)), key=lambda i: abs(table[i] - value))


def get_state_from_my_fixed_states(
        pos_observations,
        vel_observations,
        state):

    return [
        index_of(state[0], pos_observations),
        index_of(state[1], vel_observations)
    ]


def render_results(
        env,
        Q,
        pos_observations,
        vel_observations,
        actions,
        max_episode_steps=100):

    s = env.reset()
    env.render()

    for i in range(max_episode_steps):
        s = get_state_from_my_fixed_states(pos_observations, vel_observations, s)

        a = np.argmax(Q[s[0], s[1]])
        s1, _, done, _ = env.step([a])
        s = s1

        env.render()

        if done:
            break


def draw_reward_graph(rewards, iterations_count):
    plt.plot(range(iterations_count), rewards)
    plt.show()


def print_average_reward(rewards, nb_of_episodes):
    last_n_episodes = int(nb_of_episodes * 0.8)
    print(
        "Average score of last episodes: ",
        sum(rewards[-last_n_episodes:]) / last_n_episodes)


def main():
    env = gym.make('MountainCarContinuous-v0')

    pos_observations = \
        np.linspace(
            env.observation_space.low[0],
            env.observation_space.high[0],
            30)

    vel_observations = \
        np.linspace(
            -env.unwrapped.max_speed,
            env.unwrapped.max_speed,
            30)

    actions = \
        np.linspace(
            env.action_space.low[0],
            env.action_space.high[0],
            2)

    Q = np.zeros(
        [
            len(pos_observations),
            len(vel_observations),
            len(actions)
        ]
    )

    # Set learning parameters.
    learning_rate = 0.8
    discount = 0.95
    num_episodes = 420
    max_episode_steps = 2000000000

    # Create lists to contain total reward and steps per episode.
    rewards = []

    for _ in range(num_episodes):
        raw_s = env.reset()
        s = get_state_from_my_fixed_states(
            pos_observations,
            vel_observations,
            raw_s)
        total_reward = 0

        for i in range(max_episode_steps):
            # Choose an action by greedily picking from Q table with noise.
            action_index = np.argmax(
                Q[s[0], s[1]] +
                np.random.randn(1, len(actions)) * (1.0 / (i + 1)))

            a = actions[action_index]

            # Get new state and reward from env.
            raw_s1, r, done, _ = env.step([a])
            s1 = get_state_from_my_fixed_states(
                pos_observations,
                vel_observations,
                raw_s1)

            # Update Q-Table with new knowledge.
            current_action_q = Q[s[0], s[1], action_index]
            best_next_action_q = np.max(Q[s1[0], s1[1]])
            Q[s[0], s[1], action_index] = (
                current_action_q +
                learning_rate * (r + discount * best_next_action_q - current_action_q)
            )

            total_reward += r
            s = s1

            if done:
                break
            env.render()

        rewards.append(total_reward)

    # Testing
    print_average_reward(rewards, num_episodes)

    draw_reward_graph(rewards, num_episodes)
    render_results(env, Q, pos_observations, vel_observations, actions, max_episode_steps)
    env.close()


if __name__ == '__main__':
    main()
