import itertools
import numpy as np
from sklearn import linear_model
import lorenz


def compute_sindy_for_lorenz():
    clf = linear_model.Lasso(alpha=0.01, max_iter=5000)
    delta = 1e-2
    x, y, z = lorenz.get_points(dt=delta, num_steps=10000)
    c_x, c_y, c_z = x[:-1], y[:-1], z[:-1]
    p_x, p_y, p_z = x[1:], y[1:], z[1:]

    dmatrix = np.vstack([np.ones_like(c_x), c_x, c_y, c_z, c_x ** 2, c_y ** 2, c_z ** 2,
        c_x * c_y, c_y * c_z, c_x * c_z, c_x ** 2 * c_y, c_y ** 2 * c_z, c_x ** 2 * c_z])

    #noise = 0.1 * np.random.randn(*dmatrix.T.shape)
    clf.fit(dmatrix.T , p_x)#(p_x - c_x) / delta)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, p_x))


def compute_sindy(x, y, z, Vx, Vy, Vz):
    clf = linear_model.Lasso(alpha=0.01, max_iter=5000)

    f = lambda arg_x: (arg_x[1:], arg_x[:-1])
    pow_f = lambda arg_x, deg: [np.power(arg_x, i) for i in range(-deg, deg+1) if i != 0]
    pairs = lambda arg_x: [e[0] * e[1] * e[2] for e in list(itertools.combinations(arg_x, 3))]

    (p_x, c_x) = f(x)
    (p_y, c_y) = f(y)
    (p_z, c_z) = f(z)

    (p_Vx, c_Vx) = f(Vx)
    (p_Vy, c_Vy) = f(Vy)
    (p_Vz, c_Vz) = f(Vz)

    r = (c_x ** 2 + c_y ** 2 + c_z ** 2) ** 0.5
    v = (c_Vx ** 2 + c_Vy ** 2 + c_Vz ** 2) ** 0.5

    c_all = [np.ones_like(c_x), r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]

    dmatrix = np.vstack(pow_f(r, 2) + pow_f(v, 2) + pairs(c_all))

    print(f'The shape of data matrix {dmatrix.T.shape}')

    #noise = 0.1 * np.random.randn(*dmatrix.T.shape)
    for target in [p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]:
        clf.fit(dmatrix.T , target)#(p_x - c_x) / delta)
        print('Coeff')
        print(clf.coef_)
        print('Score')
        print(clf.score(dmatrix.T, target))


if __name__ == '__main__':
    compute_sindy_for_lorenz()
