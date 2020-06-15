import random
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, \
    WhiteKernel, RationalQuadratic, ConstantKernel as C
from pyeasyga import pyeasyga
from sim import generate_data, unpack_collection


KERNELS = [RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel]


def fitness(individual, data):
    x_, y_ = data

    gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=individual, n_restarts_optimizer=1, alpha=1e-6))
    scores = cross_val_score(gp, x_, y_, cv=5, scoring='r2')
    return scores.mean()


def create_individual(data):
    global KERNELS
    kernel = C(1, (1e-3, 1e3)) * KERNELS[random.randint(0, len(KERNELS)-1)]()
    return kernel


def crossover(parent_1, parent_2):
    return parent_1 * parent_2, parent_1 + parent_2 if random.randint(0, 1) == 1 else parent_1 * DotProduct()


def mutate(individual):
    global KERNELS
    individual += KERNELS[random.randint(0, len(KERNELS)-1)]()


if __name__ == '__main__':
    dt = .05

    collection = generate_data(3)
    X, Y = unpack_collection(collection, dt)

    ga = pyeasyga.GeneticAlgorithm((X, Y),
                                   population_size=8,
                                   generations=3,
                                   crossover_probability=0.8,
                                   mutation_probability=0.05,
                                   elitism=True,
                                   maximise_fitness=True)

    ga.create_individual = create_individual
    ga.crossover_function = crossover
    ga.fitness_function = fitness
    ga.mutate_function = mutate

    ga.run()
    print(ga.best_individual())
