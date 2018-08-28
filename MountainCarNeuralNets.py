import gym
import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import operator


def lerp(a, b, scalar):
    return a + (b - a) * scalar


class TfModel:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size

        # Placeholders
        self._states = None
        self._actions = None

        # Output operations
        self._output_layer = None
        self._optimizer = None
        self.var_init = None

        self._define_model()

    def _define_model(self):
        # Define the states and Q(s, a) placeholders.
        self._states = tf.placeholder(
            shape=[None, self.num_states],
            dtype=tf.float32
        )
        self._q_of_s_and_a = tf.placeholder(
            shape=[None, self.num_actions],
            dtype=tf.float32
        )

        # The layers.
        # - Create some fully connected hidden layers.
        fc1 = tf.layers.dense(self._states, units=50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, units=50, activation=tf.nn.relu)

        # - Create output layer.
        self._output_layer = tf.layers.dense(fc2, units=self.num_actions)

        # Loss & optimiser.
        loss = tf.losses.mean_squared_error(
            self._q_of_s_and_a,
            self._output_layer)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    # Get the output of the network for a given state.
    def predict_one(self, state, sess):
        return sess.run(
            self._output_layer,
            feed_dict={self._states: state.reshape(1, self.num_states)}
        )

    def predict_batch(self, states, sess):
        out = sess.run(self._output_layer, feed_dict={self._states: states})
        if math.isnan(out[0, 0]):
            raise()
        return out

    def train_batch(self, sess, in_batch, out_batch):
        sess.run(
            self._optimizer,
            feed_dict={
                self._states: in_batch,
                self._q_of_s_and_a: out_batch
            }
        )


class TrainMemory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)

        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
            pass

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(
        self,
        sess: tf.Session,
        model: TfModel,
        env: gym.Env,
        memory: TrainMemory,
        max_random_action_chance,
        min_random_action_chance,
        random_action_decay_factor,
        q_discount_factor,
        render=True
    ):
        self._sess = sess
        self._model = model
        self._env = env
        self._memory = memory
        self._max_random_action_chance = max_random_action_chance
        self._min_random_action_chance = min_random_action_chance
        self._random_action_chance = self._max_random_action_chance
        self._random_action_decay_factor = random_action_decay_factor
        self._q_discount_factor = q_discount_factor
        self._render = render

        self.reward_store = []
        self.max_x_store = []

        self._steps = 0

    def run(self):
        state = self._env.reset()
        total_reward = 0
        max_x = -100

        while True:
            if self._render:
                self._env.render()

            # Compute next state and reward.
            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)

            pos = next_state[0]
            if pos >= 0.5:
                reward += 150
            elif pos >= 0.35:
                reward += 40
            elif pos >= 0.25:
                reward += 20
            elif pos >= 0.1:
                reward += 10
            elif pos >= -0.1:
                reward += 1
            elif pos >= -0.3:
                reward += 0.001

            # Update max_x
            if next_state[0] > max_x:
                max_x = next_state[0]

            # Set to None for storage sake.
            if done:
                if pos >= 0.5:
                    reward += 5000

                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            self._steps += 1

            # Decay random action chance depending on the number of steps.
            self._random_action_chance = lerp(
                self._min_random_action_chance,
                self._max_random_action_chance,
                math.exp(-self._random_action_decay_factor * self._steps)
            )

            state = next_state
            total_reward += reward
            # print(reward, state)

            if done:
                self.reward_store.append(total_reward)
                self.max_x_store.append(max_x)
                break

        print("Step {}, total reward: {} Rand AC = {}".format(
            self._steps,
            total_reward,
            self._random_action_chance
        ))

    def _choose_action(self, state):
        if random.random() < self._random_action_chance:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    @staticmethod
    def _get_additional_reward(position):
        target = 0.5

        if position >= target:
            return 9000

        return 1 / (position - target) ** 4

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])

        next_states = []
        for val in batch:
            if val[3] is None:
                next_states.append(np.zeros(self._model.num_states))
            else:
                next_states.append(val[3])

        # Predict Q(s, a) given the batch of states.
        q_s_a = self._model.predict_batch(states, self._sess)

        # Predict Q(s', a') - so that we can do gamma * max(Q(s', a')) below.
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        # Setup training arrays.
        states_batch = np.zeros((len(batch), self._model.num_states))
        actions_batch = np.zeros((len(batch), self._model.num_actions))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]

            current_q = q_s_a[i]

            # Update the Q value for action.
            # If the game is completed, add only the reward.
            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward +\
                                    self._q_discount_factor * np.max(q_s_a_d[i])

            states_batch[i] = state
            actions_batch[i] = current_q

        self._model.train_batch(self._sess, states_batch, actions_batch)


def main():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = TfModel(num_states, num_actions, batch_size=50)
    train_memory = TrainMemory(max_memory=50000)

    with tf.Session() as sess:
        sess.run(model.var_init)

        game_runner = GameRunner(
            sess,
            model,
            env,
            train_memory,
            max_random_action_chance=0.9,
            min_random_action_chance=0.05,
            random_action_decay_factor=1/5000,
            q_discount_factor=0.99,
            render=True
        )

        num_episodes = 3000
        for episode_index in range(num_episodes):
            if episode_index % 10 == 0:
                print('Episode: {}'.format(episode_index))

            game_runner.run()

        plt.plot(game_runner.reward_store)
        plt.show()

        plt.close('all')

        plt.plot(game_runner.max_x_store)
        plt.show()


if __name__ == '__main__':
    main()
