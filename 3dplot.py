import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from sim import generate_data, train_model


if __name__ == '__main__':
    dt = .05
    tr_num = 20
    collection = generate_data(tr_num)
    gp, trajectories, targets = train_model(collection, dt)
    _, trajectories, targets = train_model(generate_data(1), dt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    v = preprocessing.minmax_scale(targets, feature_range=(0, 0.2))
    s = preprocessing.scale(trajectories)
    step = 1  # tr_num
    ax.quiver(s[::step, 0], s[::step, 1], s[::step, 2], v[::step, 0], v[::step, 1], v[::step, 2],
              color='r', arrow_length_ratio=0.3, label='target')
    ax.plot(s[::step, 0], s[::step, 1], s[::step, 2], color='b', linewidth=1)

    v = gp.predict(trajectories)
    v = preprocessing.minmax_scale(v, feature_range=(0, 0.2))
    ax.quiver(s[::step, 0], s[::step, 1], s[::step, 2], v[::step, 0], v[::step, 1], v[::step, 2],
              color='g', arrow_length_ratio=0.3, label='prediction')
    plt.legend()

    # fig = plt.figure()
    # ax = fig.add_subplot('121', projection='3d')
    # predict = gp.predict(trajectories)
    #
    # ax.scatter(s[:, 0], s[:, 2], targets[:, 1], c=targets[:, 1], cmap=cm.coolwarm)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.title.set_text('target')
    #
    # ax = fig.add_subplot('122', projection='3d')
    # ax.scatter(s[:, 0], s[:, 2], predict[:, 1], c=predict[:, 1], cmap=cm.coolwarm)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.title.set_text('prediction')

    plt.show()
