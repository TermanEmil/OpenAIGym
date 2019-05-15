import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise
from collections import deque


c_env_name = "CartPole-v1"

c_discount_rate = 0.999
c_learning_rate = 0.01

c_memory_size = 1000000
c_batch_size = 20

c_exploration_max = 1.0
c_exploration_min = 0.01
c_exploration_decay = 0.9995


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = c_exploration_max

        self.action_space = action_space
        self.memory = deque(maxlen=c_memory_size)

        self.model = tf.keras.Sequential()

        self.model.add(Dense(24, input_shape=(observation_space,), activation='relu', use_bias=True))
        self.model.add(GaussianNoise(0.05))

        self.model.add(Dense(24, activation='relu'))
        self.model.add(GaussianNoise(0.05))

        self.model.add(Dense(self.action_space, activation='linear'))

        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=c_learning_rate))
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < c_batch_size:
            return

        batch = random.sample(self.memory, c_batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward

            if not terminal:
                max_next_q = np.amax(self.model.predict(state_next))
                q_update = (reward + c_discount_rate * max_next_q)

            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.exploration_rate *= c_exploration_decay
        self.exploration_rate = max(c_exploration_min, self.exploration_rate)


def cartpole():
    env = gym.make(c_env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    target_step_reached = False

    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            # if step > 200:
            #     target_step_reached = True

            # if target_step_reached:
            env.render()

            step += 1

            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = -1 if terminal else 1

            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)

            state = state_next
            if terminal:
                print(
                    "Run: %3d, exploration: %.4f, score: %d" %
                    (run, dqn_solver.exploration_rate, step))
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()