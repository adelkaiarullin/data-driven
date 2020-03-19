import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lorenz import lorenz


def vis_trj(data, fig, label):
    ax = fig.gca(projection='3d')
    ax.plot(data[:,0], data[:,1], data[:,2], lw=1, label=label)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    ax.legend()

    # plt.show()


def get_data(start_point, num_point, delta, B, control):
    data = np.zeros((num_point, 3), np.float32)
    u_arr = np.zeros((num_point, 3), np.float32)
    for i in range(num_point):
        u = B * (control[i] - 0.001 * start_point * start_point)
        # print(u, start_point)
        u_arr[i,:] = u
        dx, dy, dz = lorenz(*start_point)
        ds = (np.array([dx, dy, dz]) + u)  * delta
        data[i,:] = start_point
        start_point += ds

    return data, u_arr        


def compute_sindy_for_lorenz(x, y, z, u, delta):
    clf = linear_model.Lasso(alpha=0.01, max_iter=5000)
    c_x, c_y, c_z = x[:-1], y[:-1], z[:-1]
    u_x, u_y, u_z = u[:-1, 0], u[:-1, 1], u[:-1, 2] 
    p_x, p_y, p_z = x[1:], y[1:], z[1:]

    dmatrix = np.vstack([np.ones_like(c_x), c_x, c_y, c_z,
        c_x * c_y, c_y * c_z, c_x * c_z, c_x ** 2 * c_y, c_y ** 2 * c_z, c_x ** 2 * c_z, 
        u_x, u_x ** 2, u_y, u_y ** 2, u_z, u_z ** 2])

    clf.fit(dmatrix.T , (p_z - c_z) / delta)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, (p_z - c_z) / delta))


if __name__ == '__main__':
    delta = .001
    t = 20
    n = int(t / delta)
    arg = np.linspace(0, t, n)

    control = 5 * np.sin(100 * arg)
    B = np.array([1., 0, -7.])
    data, u = get_data(np.array([-6., 8., 27.]), n, delta, B, control)
    compute_sindy_for_lorenz(data[:,0], data[:,1], data[:,2], u, delta)

    fig = plt.figure()
    vis_trj(data, fig, 'with control')
    B = np.array([.0, .0, .0])
    control = arg * 0
    data, u = get_data(np.array([-6., 8., 27.]), n, delta, B, control)
    vis_trj(data, fig, 'without control')
    plt.show()
