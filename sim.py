import numpy as np
import gym
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor


def compute_linear_model(x, y, u, delta):
    clf = MultiOutputRegressor(linear_model.BayesianRidge())
    c_x, c_y = x[:-1], y[:-1]
    u = u[:-1]
    p_x, p_y = x[1:], y[1:]

    dmatrix = np.vstack([np.ones_like(c_x), c_x, c_y,
                        np.sin(c_x), np.sin(c_y), np.cos(c_x), np.cos(c_y),
                        c_x * c_y, c_x ** 2, c_y ** 2,
                        c_x**3, c_y**3, u, u ** 2])

    model = ['1', 'x', 'y', 'sin x', 'sin y', 'cos x', 'cos y',
             'x*y', 'x^2', 'y^2',
             'x^3', 'y^3', 'u', 'u^2']

    target = np.stack([(p_x - c_x) / delta, (p_y - c_y) / delta], axis=-1)

    clf.fit(dmatrix.T, target)
    print('Score')
    print(clf.score(dmatrix.T, target))

    for coor, e in zip(['x', 'y', 'z'], clf.estimators_):
        print(f'Coefficients {coor}')
        print([(a, b) for (a, b) in zip(model, e.coef_)])


env = gym.make('Pendulum-v0')
env.reset()

num_iter = 200
data = np.zeros(shape=(num_iter, 3), dtype=np.float32)

for step in range(num_iter):
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)  # take a random action
    data[step, 0] = np.arccos(observation[0])
    data[step, 1] = observation[-1]
    data[step, -1] = action

    print(f'Step {step}\nObservation {observation}\nReward {reward}\nAction {action}')

env.close()

data += np.random.randn(*data.shape) * 0.01
dt = .05
compute_linear_model(data[:, 0], data[:, 1], data[:, 2], dt)
