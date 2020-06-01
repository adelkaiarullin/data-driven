import matplotlib.pyplot as plt
import time
import numpy as np
from sim import train_model, generate_data


if __name__ == '__main__':
    dt = .05
    observation = []
    training_time = []

    for i in range(1, 3):
        collection = generate_data(i)
        start = time.time()
        gp, dm, tag = train_model(collection, dt)
        end = time.time()
        training_time.append(end - start)
        observation.append(gp.score(dm, tag))

    plt.figure()
    plt.plot(range(1, len(observation) + 1), observation, 'm', lw=2, zorder=9)
    plt.xlabel('number of trajectories')
    plt.ylabel('R2 score')
    plt.tight_layout()

    plt.figure()
    plt.plot(range(1, len(training_time) + 1), training_time, 'm', lw=2, zorder=9, label='experiment curve')
    x = np.array(range(1, len(training_time) + 1))
    y = np.array(training_time)
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(x, p(x), 'b', lw=3, zorder=9, label='fitting curve')
    plt.xlabel('number of trajectories')
    plt.ylabel('Training time')
    plt.legend()
    plt.tight_layout()
    plt.show()
