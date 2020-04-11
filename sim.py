import numpy as np
import gym
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm


def unpack_collection(obs):
    in_x = []
    out_x = []
    in_y = []
    out_y = []
    in_z = []
    out_z = []
    in_u = []

    for d in obs:
        x, y, z, u = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        in_x.append(x[:-1])
        in_y.append(y[:-1])
        in_z.append(z[:-1])
        out_x.append(x[1:])
        out_y.append(y[1:])
        out_z.append(z[1:])
        in_u.append(u[:-1])

    in_x = np.concatenate(in_x)
    in_y = np.concatenate(in_y)
    in_z = np.concatenate(in_z)
    in_u = np.concatenate(in_u)
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    out_z = np.concatenate(out_z)

    return in_x, in_y, in_z, in_u, out_x, out_y, out_z


def compute_model(obs, delta):
    c_x, c_y, c_z, u, n_x, n_y, n_z = unpack_collection(obs)

    kernel = C(10.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel))
    br = MultiOutputRegressor(BayesianRidge())
    data_matrix = np.vstack([c_x, c_y, c_z, u]).T
    target = np.stack([(n_x - c_x) / delta, (n_y - c_y) / delta, (n_z - c_z) / delta], axis=-1)

    data_matrix += np.random.randn(*data_matrix.shape) * 1e-3  # make some noise
    data_matrix = preprocessing.scale(data_matrix)
    target = preprocessing.scale(target)
    target += np.random.randn(*target.shape) * 1e-2  # make some noise

    scores = cross_val_score(br, data_matrix, target, cv=5, scoring='r2')
    print("BR: coefficient of determination: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))

    scores = cross_val_score(gp, data_matrix, target, cv=5, scoring='r2')
    print("GP: coefficient of determination: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))


def generate_data(epochs=10):
    assert epochs > 0
    env = gym.make('Pendulum-v0')
    collection_ = []

    for _ in tqdm(range(epochs)):
        data = []
        env.reset()
        while True:
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)  # take a random action
            data.append(np.append(observation, action))
            if done:
                break

        collection_.append(np.array(data))

    env.close()

    return collection_


if __name__ == '__main__':
    collection = generate_data(20)
    dt = .05
    compute_model(collection, dt)
