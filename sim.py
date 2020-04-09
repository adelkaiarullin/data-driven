import numpy as np
import gym
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor


def compute_model(obs, delta):
    in_x = []
    out_x = []
    in_y = []
    out_y = []
    in_u = []

    for d in obs:
        x, y, u = d[:, 0], d[:, 1], d[:, 2]
        in_x.append(x[:-1])
        in_y.append(y[:-1])
        out_x.append(x[1:])
        out_y.append(y[1:])
        in_u.append(u[:-1])

    c_x = np.concatenate(in_x)
    c_y = np.concatenate(in_y)
    u = np.concatenate(in_u)
    p_x = np.concatenate(out_x)
    p_y = np.concatenate(out_y)

    reg = MultiOutputRegressor(GaussianProcessRegressor())
    data_matrix = np.vstack([np.ones_like(c_x), c_x, c_y, np.sin(c_x), np.sin(c_y), np.cos(c_x), np.cos(c_y), u]).T
    target = np.stack([(p_x - c_x) / delta, (p_y - c_y) / delta], axis=-1)

    reg.fit(data_matrix, target)
    print(f'Score {reg.score(data_matrix, target)}')


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.reset()

    epochs = 20
    num_iter = 200
    collection = []

    for epoch in range(epochs):
        data = np.zeros(shape=(num_iter, 3), dtype=np.float32)
        for step in range(num_iter):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)  # take a random action
            data[step, 0] = np.arccos(observation[0])
            data[step, 1] = observation[-1]
            data[step, -1] = action

            print(f'Epoch {epoch+1}\nStep {step+1}\nObservation {observation}\nReward {reward}\nAction {action}')

        data += np.random.randn(*data.shape) * 1e-3
        collection.append(data)

    env.close()

    dt = .05
    compute_model(collection, dt)
