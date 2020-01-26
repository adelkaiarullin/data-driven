import itertools
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import idds
import lorenz


def get_predicted_track(clfs, x, step, dt):
    x = x.reshape(1, 6)
    track = [x]
    k = 0
    while k < step:
        state = np.zeros(shape=(1, 6), dtype=np.float32)
        for i, (name, clf) in enumerate(zip(['x', 'y', 'z', 'vx', 'vy', 'vz'], clfs)):
            dm = get_dynamics(name, x[0][0],x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], dt[k])
            print(f'Dm {dm} \n{dm.shape}')
            s = clf.predict(dm)
            state[0][i] = s
        x += state
        track.append(x)
        k += 1

    return track



def get_dynamics(name, x, y, z, vx, vy, vz, dt):
    v = np.sqrt(vx * vx + vy * vy + vz * vz)
    r = np.sqrt(x * x + y * y + z * z)

    if name == 'x':
        return np.vstack([dt*vx, r*x, r*vx, v*x, v*vx, dt*r*vx, dt*v*vx, vx, r*v*vx, vx]).T
    elif name == 'y':
        return np.vstack([dt*vy, r*y, r*vy, v*y, v*vy, dt*r*vy, dt*v*vy, vy, r*v*vy, vy]).T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    elif name == 'z':
        return np.vstack([dt*vz, r*z, r*vz, v*z, v*vz, dt*r*vz, dt*v*vz, vz, r*v*vz, vz]).T
    elif name == 'vx':
        return np.vstack([dt*x, dt*vx, r*x, v*x, dt*x/r, dt*v*x, dt*v*vx, r*v*vx]).T
    elif name == 'vy':
        return np.vstack([dt*y, dt*vy, r*y, v*y, dt*y/r, dt*v*y, dt*v*vy, r*v*vy]).T
    elif name == 'vz':
        return np.vstack([dt*z, dt*vz, r*z, v*z, dt*z/r, dt*v*z, dt*v*vz, r*v*vz]).T



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
    clf.fit(dmatrix.T , (p_z - c_z) / delta)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, (p_z - c_z) / delta))

def compute_dynamics(x, y, z, Vx, Vy, Vz, sx, sy, sz, sVx, sVy, sVz, t):
#def compute_dynamics(x, y, z, Vx, Vy, Vz, t):
    f = lambda arg_x: (arg_x[1:], arg_x[:-1])

    (p_t, c_t) = f(t)
    dt = np.zeros_like(x)
    dt[1:] = p_t - c_t
    (p_x, c_x) = x, sx#f(x)
    (p_y, c_y) = y, sy#f(y)
    (p_z, c_z) = z, sz#f(z)

    (p_Vx, c_Vx) = Vx, sVx#f(Vx)
    (p_Vy, c_Vy) = Vy, sVy#f(Vy)
    (p_Vz, c_Vz) = Vz, sVz#f(Vz)

    arr_p = []
    #clfs = []
    for target, name in zip([p_x - c_x, p_y - c_y, p_z - c_z, p_Vx - c_Vx, p_Vy - c_Vy, p_Vz - c_Vz], ['x', 'y', 'z', 'vx', 'vy', 'vz']):
        dmatrix = get_dynamics(name, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, dt)
        dmatrix -= dmatrix.mean(axis=0)
        dmatrix /= dmatrix.std(axis=0)
        clf = linear_model.LinearRegression(n_jobs=-1)
        clf.fit(dmatrix , target)
        #clfs.append(clf)
        arr_p.append(clf.predict(dmatrix))
        #print(f'Coeff {name}\n{clf.coef_}')
    
    #track = get_predicted_track(clfs, np.array([c_x[0], c_y[0], c_z[0], c_Vx[0], c_Vy[0], c_Vz[0]]), p_x.shape[0] - 1, p_t-c_t)
    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    predict = np.vstack(arr_p).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    sm = 100 * (1 - idds.smape(target, predict + source))
    print(f'SMAPE {sm}')
    st_sm = 100 * (1 - idds.smape(target, source))
    print(f'SMAPE {st_sm}')
     
    return sm


def compute_sindy(x, y, z, Vx, Vy, Vz, t):
#def compute_sindy(x, y, z, Vx, Vy, Vz, sx, sy, sz, sVx, sVy, sVz, t):
    f = lambda arg_x: (arg_x[1:], arg_x[:-1])
    comb = lambda arg_x: [e[0] * e[1] *e[2] for e in list(itertools.combinations(arg_x, 3))]

    (p_t, c_t) = f(t)
    #dt = np.zeros_like(x)
    dt = p_t - c_t
    (p_x, c_x) = f(x)
    (p_y, c_y) = f(y)
    (p_z, c_z) = f(z)

    (p_Vx, c_Vx) = f(Vx)
    (p_Vy, c_Vy) = f(Vy)
    (p_Vz, c_Vz) = f(Vz)
    
    r = np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)
    v = np.sqrt(c_Vx ** 2 + c_Vy ** 2 + c_Vz ** 2)
    
    c_all = [np.ones_like(r), dt, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]#best
    dmatrix = np.vstack(comb(c_all) + [dt, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz])

    #print(f'The shape of data matrix {dmatrix.T.shape}')
    dmatrix = dmatrix.T# + 0.1 * np.random.randn(*dmatrix.T.shape)
    coeff = []
    intercept = []
    #dmatrix -= dmatrix.mean(axis=0)
    #dmatrix /= dmatrix.std(axis=0)

    for target in [p_x - c_x, p_y - c_y, p_z - c_z, p_Vx - c_Vx, p_Vy - c_Vy, p_Vz - c_Vz]:
        clf = linear_model.Lasso(alpha=1e-2, max_iter=1000)
        #clf = linear_model.Ridge()
        #clf = linear_model.HuberRegressor(max_iter=10)
        clf.fit(dmatrix , target)
        coeff.append(clf.coef_)
        intercept.append(clf.intercept_)
        #print(clf.coef_)
        #print(clf.intercept_)

    coeff = np.vstack(coeff).T
    intercept = np.array(intercept).reshape(1, 6)
    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    sm = 100 * (1 - idds.smape(target, dmatrix.dot(coeff) + intercept + source))
    #print(f'Coeff \n{coeff}')
    #sm = 100 * (1 - idds.smape(target, clf.predict(coeff) ))
    print(f'SMAPE {sm}')
    print(f'SMAPE {100 * (1 - idds.smape(target, source))}')

    # split = np.array(test[:, -6:], dtype=np.float32)
    # t_x, t_y, t_z, t_Vx, t_Vy, t_Vz = split[:, 0], split[:, 1], split[:, 2], split[:, 3], split[:, 4], split[:, 5]
    # tr = np.sqrt(t_x ** 2 + t_y ** 2 + t_z ** 2)
    # tv = np.sqrt(t_Vx ** 2 + t_Vy ** 2 + t_Vz ** 2)
    # t_all = [np.ones_like(tr), 1/tr, tr, tv, t_x, t_y, t_z, t_Vx, t_Vy, t_Vz, np.sin(tr), np.sin(tv)]
    # t_all = comb(t_all) + [ 1/tr, tr, tv, t_x, t_y, t_z, t_Vx, t_Vy, t_Vz, np.sin(tr), np.sin(tv)]
    # dmatrix = np.vstack(t_all).T
    # dmatrix -= dmatrix.mean(axis=0)
    # dmatrix /= dmatrix.std(axis=0)

    # save_submission(np.concatenate([test[:, 0].reshape(len(t_x), 1), dmatrix.dot(coeff) + intercept], axis=1))
     
    return sm, coeff




if __name__ == '__main__':
    compute_sindy_for_lorenz()
