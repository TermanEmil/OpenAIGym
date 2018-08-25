import gym
import numpy as np
import matplotlib.pyplot as plt


def render_results(env, Q, max_episode_steps=100):
    s = env.reset()
    env.render()

    for i in range(max_episode_steps):
        a = np.argmax(Q[s, :])
        s1, _, done, _ = env.step(a)
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
    env = gym.make('Taxi-v2')

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Set learning parameters.
    learning_rate = 0.8
    discount = 1
    num_episodes = 500
    max_episode_steps = 1000

    # Create lists to contain total reward and steps per episode.
    rewards = []

    for _ in range(num_episodes):
        s = env.reset()
        total_reward = 0

        for _ in range(max_episode_steps):
            # Choose an action by greedily picking from Q table.
            a = np.argmax(Q[s])

            # Get new state and reward from env.
            s1, r, done, _ = env.step(a)

            # Update Q-Table with new knowledge.
            Q[s, a] = (
                Q[s, a] +
                learning_rate * (r + discount * np.max(Q[s1]) - Q[s, a])
            )

            total_reward += r
            s = s1

            if done:
                break

        rewards.append(total_reward)

    # Testing
    render_results(env, Q, max_episode_steps)
    print_average_reward(rewards, num_episodes)

    draw_reward_graph(rewards, num_episodes)


if __name__ == '__main__':
    main()