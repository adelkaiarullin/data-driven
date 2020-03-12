import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lorenz import lorenz


def vis_trj(data):
    ax = fig.gca(projection='3d')
    ax.plot(data[:,0], data[:,1], data[:,2], lw=1)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    # plt.show()


def get_data(start_point, num_point, delta, control):
    data = np.zeros((num_point, 3), np.float32)
    for i in range(num_point):
        dx, dy, dz = lorenz(*(start_point + control[i]))
        ds = np.array([dx, dy, dz]) * delta 
        data[i] = start_point
        start_point += ds

    return data        


def compute_sindy_for_lorenz():
    clf = linear_model.Lasso(alpha=0.01, max_iter=5000)
    delta = 1e-2
    x, y, z = lorenz.get_points(dt=delta, num_steps=10000)
    c_x, c_y, c_z = x[:-1], y[:-1], z[:-1]
    p_x, p_y, p_z = x[1:], y[1:], z[1:]

    dmatrix = np.vstack([np.ones_like(c_x), c_x, c_y, c_z, c_x ** 2, c_y ** 2, c_z ** 2,
        c_x * c_y, c_y * c_z, c_x * c_z, c_x ** 2 * c_y, c_y ** 2 * c_z, c_x ** 2 * c_z])

    clf.fit(dmatrix.T , (p_z - c_z) / delta)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, (p_z - c_z) / delta))


if __name__ == '__main__':
    n = 500
    t = 10
    delta = t / n
    arg = np.linspace(0, t, n)
    control1 = arg * 0
    control = 5 * np.sin(20 * arg)
    data = get_data(np.array([-8., 8., 27.]), n, delta, control)

    fig = plt.figure()
    vis_trj(data)
    data = get_data(np.array([-8., 8., 27.]), n, delta, control1)
    vis_trj(data)
    plt.show()
