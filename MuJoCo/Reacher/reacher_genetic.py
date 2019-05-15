import gym
import tensorflow as tf
import random
from deap import tools
from keras_genetic_tools import *


env_name = 'Reacher-v2'

nb_of_generations = 50
nb_of_genomes = 50

# Evolution
percentage_to_reinsert = 0.1
percentage_to_cross = 0.6


class Brain:
    def __init__(self, keras_model, nb_of_weights_per_layer):
        self.model = keras_model
        self.nb_of_weights_per_layer = nb_of_weights_per_layer

    def clone(self):
        cloned_model = tf.keras.models.clone_model(self.model)
        cloned_model.set_weights(self.model.get_weights())
        return Brain(cloned_model, self.nb_of_weights_per_layer)

    def predict_action(self, observations):
        return self.model.predict(observations)

    def get_trainable_vars(self):
        weights = extract_flat_weights_from_keras_model(self.model)
        return weights

    def set_trainable_vars(self, weights):
        assert len(weights) == sum(self.nb_of_weights_per_layer)
        apply_weights_to_keras_model(
            weights,
            self.model,
            self.nb_of_weights_per_layer)


class Genome:
    def __init__(self, brains):
        assert isinstance(brains, Brain)

        self.brains = brains
        self.fitness = 0

    def clone(self):
        return Genome(self.brains.clone())

    def evaluate(self, env, render=False):
        total_reward = 0

        env.seed(42)
        state = env.reset()

        while True:
            action = self.brains.predict_action([[state]])[0]
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

            if render:
                env.render()

            if done:
                break

        self.fitness = total_reward


class MutationManager:
    def __init__(self, mutations):
        self.mutations = mutations

    def mutate_vars(self, variables):
        for i in range(len(variables)):
            var = variables[i]
            for mutation in self.mutations:
                var = mutation(var)

            variables[i] = var

    def mutate_brain(self, brain):
        assert isinstance(brain, Brain)

        trainable_vars = brain.get_trainable_vars()
        self.mutate_vars(trainable_vars)
        brain.set_trainable_vars(trainable_vars)


class Population:
    def __init__(
            self,
            env_name,
            genome_model,
            mutation_manager,
            crossover,
            nb_of_genomes=50):

        assert isinstance(genome_model, Genome)
        assert isinstance(mutation_manager, MutationManager)

        self.env_name = env_name
        self.genome_model = genome_model
        self.mutation_manager = mutation_manager
        self.crossover = crossover

        self.nb_of_genomes = nb_of_genomes

        self.genomes = []
        self.environments = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for env in self.environments:
            env.close()

    def init_genomes(self):
        for _ in range(self.nb_of_genomes):
            self.genomes.append(self.genome_model.clone())

    def init_environments(self):
        env = gym.make(self.env_name)
        self.environments.append(env)

    def evaluate(self):
        for i in range(self.nb_of_genomes):
            genome = self.genomes[i]
            env = self.environments[0]

            genome.evaluate(env)
            # print(genome.fitness)

    def crossover_genomes(self, g1, g2):
        assert isinstance(g1, Genome)
        assert isinstance(g2, Genome)

        w1 = g1.brains.get_trainable_vars()
        w2 = g2.brains.get_trainable_vars()

        child1_w, child2_w = self.crossover(w1, w2)

        child1 = g1.clone()
        child2 = g2.clone()

        child1.brains.set_trainable_vars(child1_w)
        child2.brains.set_trainable_vars(child2_w)

        return child1, child2

    def evolve(self):
        new_generation = []

        sorted_genomes = sorted(
            self.genomes,
            key=lambda x: x.fitness,
            reverse=True)

        best_genome = sorted_genomes[0]
        print([x.fitness for x in sorted_genomes])

        nb_of_genomes_to_reinsert = int(self.nb_of_genomes * percentage_to_reinsert)

        nb_of_genomes_to_cross = int(self.nb_of_genomes * percentage_to_cross)
        if nb_of_genomes_to_cross % 2 == 1:
            nb_of_genomes_to_cross += 1

        nb_of_genomes_to_mutate =\
            self.nb_of_genomes -\
            nb_of_genomes_to_reinsert -\
            nb_of_genomes_to_cross

        for genome in sorted_genomes[0:nb_of_genomes_to_reinsert]:
            new_generation.append(genome.clone())

        for genome in sorted_genomes[0:nb_of_genomes_to_mutate]:
            genome = genome.clone()
            self.mutation_manager.mutate_brain(genome.brains)
            new_generation.append(genome)

        for i in range(0, nb_of_genomes_to_cross, 2):
            parent1, parent2 = sorted_genomes[i], sorted_genomes[i + 1]
            child1, child2 = self.crossover_genomes(parent1, parent2)

            # self.mutation_manager.mutate_brain(child1.brains)
            # self.mutation_manager.mutate_brain(child2.brains)

            new_generation.append(child1)
            new_generation.append(child2)

        assert len(new_generation) == self.nb_of_genomes
        self.genomes = new_generation

        best_genome.evaluate(self.environments[0], True)


g_mut_scale = 1.0


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def mut_scalar_add_uniform(value, scalar, chance):
    chance *= g_mut_scale

    if random.random() < chance:
        scalar *= g_mut_scale
        value = clamp(value + random.uniform(-scalar, scalar), -3, 3)
        print(value)

    return value


def mut_random_value(value, bound, chance):
    chance *= g_mut_scale

    if random.random() < chance:
        value = random.uniform(-bound, bound)

    return value


def get_nb_of_inputs_and_outputs_of_env(env_name):
    with gym.make(env_name) as env:
        nb_of_inputs = env.observation_space.shape[0]
        nb_of_outputs = env.action_space.shape[0]

        return nb_of_inputs, nb_of_outputs


def main():
    nb_of_inputs, nb_of_outputs = get_nb_of_inputs_and_outputs_of_env(env_name)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2 ** 8, input_shape=(nb_of_inputs,), activation='tanh', use_bias=True, name='dense_1'),
        # tf.keras.layers.Dense(2**8, activation='tanh', name='dense_2', use_bias=True),
        tf.keras.layers.Dense(nb_of_outputs, activation='tanh', name='output'),
    ])

    nb_of_weights_per_layer = extract_nb_of_weights_per_layer(model)
    brain = Brain(model, nb_of_weights_per_layer)
    genome_model = Genome(brain)

    crossover = lambda p1, p2: tools.cxOnePoint(p1, p2)

    mutations = [
        lambda v: mut_scalar_add_uniform(v, 1., 0.2),
        lambda v: mut_scalar_add_uniform(v, 1, 0.01),
        lambda v: mut_random_value(v, 1, 0.02),
    ]

    mutation_manager = MutationManager(mutations)

    population = Population(
        env_name=env_name,
        genome_model=genome_model,
        crossover=crossover,
        mutation_manager=mutation_manager,
        nb_of_genomes=nb_of_genomes)

    with population as population:
        population.init_genomes()
        population.init_environments()

        for i in range(nb_of_generations):
            global g_mut_scale

            population.evaluate()
            population.evolve()

            g_mut_scale = (1. / 1.3) ** i


if __name__ == '__main__':
    main()


