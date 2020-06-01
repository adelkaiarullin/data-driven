import matplotlib.pyplot as plt
import numpy as np
from sim import train_model, generate_data, unpack_collection


def integrate(gp, data, dt):
    entry_point = data[0, :].reshape(1, 4)
    num_points = len(data)
    x = [entry_point[0, 0]]
    y = [entry_point[0, 1]]
    z = [entry_point[0, 2]]

    for i in range(num_points - 1):
        v = gp.predict(entry_point)
        x.append(x[-1] + v[0, 0] * dt)
        y.append(y[-1] + v[0, 1] * dt)
        z.append(z[-1] + v[0, 2] * dt)

        entry_point[0, 0] = x[-1]
        entry_point[0, 1] = y[-1]
        entry_point[0, 2] = z[-1]
        entry_point[0, 3] = data[i + 1, -1]

    return np.array([x, y, z]).T


if __name__ == '__main__':
    dt = .05

    collection = generate_data(30)
    gp, _, _ = train_model(collection, dt)
    collection = generate_data(1)
    data, _ = unpack_collection(collection, dt)

    p = integrate(gp, data, dt)

    plt.figure(1)
    plt.subplot('121')
    plt.plot(range(1, len(data) + 1), data[:, 0], '--', linewidth=3, label='True x1')
    plt.plot(range(1, len(p) + 1), p[:, 0], label='Predicted x')
    plt.legend()
    plt.subplot('122')
    plt.plot(range(1, len(p) + 1), data[:, 0] - p[:, 0], label='Error', color='r')
    plt.legend()

    plt.figure(2)
    plt.subplot('121')
    plt.plot(range(1, len(data) + 1), data[:, 1], '--', linewidth=3, label='True x2')
    plt.plot(range(1, len(p) + 1), p[:, 1], label='Predicted x')
    plt.legend()
    plt.subplot('122')
    plt.plot(range(1, len(p) + 1), data[:, 1] - p[:, 1], label='Error', color='r')
    plt.legend()

    plt.figure(3)
    plt.subplot('121')
    plt.plot(range(1, len(data) + 1), data[:, 2], '--', linewidth=3, label='True x3')
    plt.plot(range(1, len(p) + 1), p[:, 2], label='Predicted x')
    plt.legend()
    plt.subplot('122')
    plt.plot(range(1, len(p) + 1), data[:, 2] - p[:, 2], label='Error', color='r')
    plt.tight_layout()
    plt.legend()
    plt.show()
