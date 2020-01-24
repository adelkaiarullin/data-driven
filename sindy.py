import itertools
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import idds
import lorenz


def save_submission(arr):
    np.savetxt('submission.csv', arr, delimiter=',', fmt='%d,%f,%f,%f,%f,%f,%f', header='id,x,y,z,Vx,Vy,Vz')

def compute_sindy_for_lorenz():
    clf = linear_model.Lasso(alpha=0.01, max_iter=5000)
    delta = 1e-2
    x, y, z = lorenz.get_points(dt=delta, num_steps=10000)
    c_x, c_y, c_z = x[:-1], y[:-1], z[:-1]
    p_x, p_y, p_z = x[1:], y[1:], z[1:]

    dmatrix = np.vstack([np.ones_like(c_x), c_x, c_y, c_z, c_x ** 2, c_y ** 2, c_z ** 2,
        c_x * c_y, c_y * c_z, c_x * c_z, c_x ** 2 * c_y, c_y ** 2 * c_z, c_x ** 2 * c_z])

    #noise = 0.1 * np.random.randn(*dmatrix.T.shape)
    clf.fit(dmatrix.T , p_x)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, p_x))


#def compute_sindy(x, y, z, Vx, Vy, Vz, t):
def compute_sindy(x, y, z, Vx, Vy, Vz, sx, sy, sz, sVx, sVy, sVz, t, test):
    f = lambda arg_x: (arg_x[1:], arg_x[:-1])

    pow_f = lambda arg_x, deg: [np.power(arg_x, i) for i in range(-deg, deg + 1) if i != 0]
    comb = lambda arg_x: [e[0] * e[1] *e[2] for e in list(itertools.combinations(arg_x, 3))]

    #(p_t, c_t) = f(t)
    (p_x, c_x) = x, sx#f(x)
    (p_y, c_y) = y, sy#f(y)
    (p_z, c_z) = z, sz#f(z)

    (p_Vx, c_Vx) = Vx, sVx#f(Vx)
    (p_Vy, c_Vy) = Vy, sVy#f(Vy)
    (p_Vz, c_Vz) = Vz, sVz#f(Vz)
    
    r = np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)
    v = np.sqrt(c_Vx ** 2 + c_Vy ** 2 + c_Vz ** 2)
    
    #c_all = [np.ones_like(r), t, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)]#best
    c_all = [np.ones_like(r), 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)]

    #dmatrix = np.vstack(comb(c_all) + [ t, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)])
    dmatrix = np.vstack(comb(c_all) + [ 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)])

    #print(f'The shape of data matrix {dmatrix.T.shape}')
    dmatrix = dmatrix.T# + 0.1 * np.random.randn(*dmatrix.T.shape)
    coeff = []
    intercept = []
    dmatrix -= dmatrix.mean(axis=0)
    dmatrix /= dmatrix.std(axis=0)

    for target in [p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]:
        clf = linear_model.Lasso(alpha=1e-3, max_iter=5000)
        #clf = linear_model.HuberRegressor(max_iter=200)
        clf.fit(dmatrix , target)
        coeff.append(clf.coef_)
        intercept.append(clf.intercept_)
        #print(clf.coef_)
        print(clf.intercept_)

    coeff = np.vstack(coeff).T
    intercept = np.array(intercept).reshape(1, 6)
    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    sm = 100 * (1 - idds.smape(target, dmatrix.dot(coeff) + intercept))
    #print(f'Coeff \n{coeff}')
    #sm = 100 * (1 - idds.smape(target, clf.predict(coeff) ))
    print(f'SMAPE {sm}')
    print(f'SMAPE {100 * (1 - idds.smape(target, source))}')

    split = np.array(test[:, -6:], dtype=np.float32)
    t_x, t_y, t_z, t_Vx, t_Vy, t_Vz = split[:, 0], split[:, 1], split[:, 2], split[:, 3], split[:, 4], split[:, 5]
    tr = np.sqrt(t_x ** 2 + t_y ** 2 + t_z ** 2)
    tv = np.sqrt(t_Vx ** 2 + t_Vy ** 2 + t_Vz ** 2)
    t_all = [np.ones_like(tr), 1/tr, tr, tv, t_x, t_y, t_z, t_Vx, t_Vy, t_Vz, np.sin(tr), np.sin(tv)]
    t_all = comb(t_all) + [ 1/tr, tr, tv, t_x, t_y, t_z, t_Vx, t_Vy, t_Vz, np.sin(tr), np.sin(tv)]
    dmatrix = np.vstack(t_all).T
    dmatrix -= dmatrix.mean(axis=0)
    dmatrix /= dmatrix.std(axis=0)

    save_submission(np.concatenate([test[:, 0].reshape(len(t_x), 1), dmatrix.dot(coeff) + intercept], axis=1))
     
    return sm, coeff




if __name__ == '__main__':
    compute_sindy_for_lorenz()
