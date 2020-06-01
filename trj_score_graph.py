import matplotlib.pyplot as plt
import numpy as np
from sim import compute_model, generate_data


if __name__ == '__main__':
    dt = .05
    observation = []

    for i in range(1, 26):
        collection = generate_data(i)
        scores = compute_model(collection, dt)
        observation.append((scores.mean(), scores.std()))

    y_mean = np.array([e[0] for e in observation])
    y_std = np.array([e[1] for e in observation])
    plt.plot(range(1, len(y_mean) + 1), y_mean, 'm', lw=3, zorder=9)
    plt.fill_between(range(1, len(y_mean) + 1), y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='b')
    plt.title('trajectories - cv score', fontsize=12)
    plt.tight_layout()
    plt.show()
