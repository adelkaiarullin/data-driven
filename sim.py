import numpy as np
import gym
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C, DotProduct
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm


def unpack_collection(obs, dt):
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

    data_matrix = np.vstack([in_x, in_y, in_z, in_u]).T
    target = np.stack([(out_x - in_x) / dt, (out_y - in_y) / dt, (out_z - in_z) / dt], axis=-1)

    return data_matrix, target


def compute_model(obs, delta, kernel=None):
    if kernel is None:
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic()

    gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-6))
    br = MultiOutputRegressor(BayesianRidge())

    data_matrix, target = unpack_collection(obs, delta)

    scores = cross_val_score(br, data_matrix, target, cv=5, scoring='r2')
    print("BR: coefficient of determination: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))

    scores = cross_val_score(gp, data_matrix, target, cv=5, scoring='r2')
    print("GP: coefficient of determination: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))

    return scores


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


def train_model(obs, delta, kernel_f=None):
    data_matrix, target = unpack_collection(obs, delta)

    if kernel_f is None:
        kernel = C(1, (1e-3, 1e3)) * RationalQuadratic()
    else:
        kernel = kernel_f

    gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-6))
    train_x, test_x, train_y, test_y = train_test_split(data_matrix, target, test_size=0.8)

    gp.fit(train_x, train_y)
    score = gp.score(test_x, test_y)
    print(f'Score {score}')

    return gp, data_matrix, target


if __name__ == '__main__':
    collection = generate_data(3)
    dt = .05
    compute_model(collection, dt)
