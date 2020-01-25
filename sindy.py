import itertools
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import idds
import lorenz


def get_predicted_track(clf, x, step):
    track = [x]

    while step > 0:
        step -= 1
        state = np.zeros(shape=(1, 6), dtype=np.float32)
        for i, name in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz']):
            state[0, i] = clf.predict(get_dynamics(name, *x))
        x = state
        track.append(state)

    return track



def get_dynamics(name, x, y, z, vx, vy, vz, dt):
    v = np.sqrt(vx * vx + vy * vy + vz * vz)
    r = np.sqrt(x * x + y * y + z * z)

    if name == 'x':
        return np.vstack([dt*x, r*x, x/r, vx/r, v*vx, dt*vx/r, dt*v*x, x, vx, r*v*x, v*x/r, v*vx/r, r*x*np.sin(v), vx]).T
    elif name == 'y':
        return np.vstack([dt*y, dt*vy, r*y, y/r, vy/r, v*vy, dt*y/r, dt*vy/r, dt*v*y, dt*y*np.sin(v), y, vy, r*v*y, r*y*np.sin(v), v*y/r, v*vy/r, y*np.sin(v)/r, y, vy]).T
    elif name == 'z':
        return np.vstack([dt*z, dt*vz, r*z, z/r, vz*z/r, vz/r, v*z, v*vz, dt*r*v, dt*r*vz, dt*z/r, dt*vz/r, dt*v*z, dt*v*vz, z, vz, dt*z*np.sin(v), r*v*z, r*z*np.sin(v)]).T
    elif name == 'vx':
        return np.vstack([dt*vx, vx/r, v*x, dt*r*x, dt*vx/r, dt*v*x, dt*v*vx, dt*vx*np.sin(v), v*vx/r, vx*np.sin(v)/r,v*x*np.sin(v), x]).T
    elif name == 'vy':
        return np.vstack([dt*y, dt*vy, vy/r, v*y, dt*vy/r, dt*v*y, dt*v*vy, dt*vy*np.sin(v), v*vy/r, v*y*np.sin(v), v*np.sin(v)/r, y]).T
    elif name == 'vz':
        return np.vstack([dt*z, dt*vz, vz/r, v*z, v*vz, dt*r*z, dt*vz/r, dt*v*z, dt*v*vz, dt*vz*np.sin(v), z, v*vz/r, vz*np.sin(v)/r, v*z*np.sin(v)]).T



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



def compute_dynamics(x, y, z, Vx, Vy, Vz, t):
    f = lambda arg_x: (arg_x[1:], arg_x[:-1])
    (p_t, c_t) = f(t)
    (p_x, c_x) = f(x)
    (p_y, c_y) = f(y)
    (p_z, c_z) = f(z)

    (p_Vx, c_Vx) = f(Vx)
    (p_Vy, c_Vy) = f(Vy)
    (p_Vz, c_Vz) = f(Vz)

    arr_p = []
    for target, name in zip([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz], ['x', 'y', 'z', 'vx', 'vy', 'vz']):
        dmatrix = get_dynamics(name, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, p_t - c_t)
        #clf = linear_model.Lasso(alpha=1e-3, max_iter=1000)
        clf = linear_model.Ridge()
        #clf = linear_model.HuberRegressor(max_iter=200)
        clf.fit(dmatrix , target)
        track = get_predicted_track(clf, np.array([c_x[0], c_y[0], c_z[0], c_Vx[0], c_Vy[0], c_Vz[0], (p_t - c_t)[0]]), p_x.shape[0] - 1)
        arr_p.append(clf.predict(dmatrix))
            
    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    predict = np.vstack(arr_p).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    sm = 100 * (1 - idds.smape(target, predict))
    #print(f'Coeff \n{coeff}')
    #sm = 100 * (1 - idds.smape(target, clf.predict(coeff) ))
    print(f'SMAPE {sm}')
    print(f'SMAPE {100 * (1 - idds.smape(target, source))}')
     
    return sm


def compute_sindy(x, y, z, Vx, Vy, Vz, t):
#def compute_sindy(x, y, z, Vx, Vy, Vz, sx, sy, sz, sVx, sVy, sVz, t):
    f = lambda arg_x: (arg_x[1:], arg_x[:-1])
    comb = lambda arg_x: [e[0] * e[1] *e[2] for e in list(itertools.combinations(arg_x, 3))]

    (p_t, c_t) = f(t)
    (p_x, c_x) = f(x)
    (p_y, c_y) = f(y)
    (p_z, c_z) = f(z)

    (p_Vx, c_Vx) = f(Vx)
    (p_Vy, c_Vy) = f(Vy)
    (p_Vz, c_Vz) = f(Vz)
    
    r = np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)
    v = np.sqrt(c_Vx ** 2 + c_Vy ** 2 + c_Vz ** 2)
    
    c_all = [np.ones_like(r), p_t-c_t, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)]#best
    dmatrix = np.vstack(comb(c_all) + [p_t-c_t, 1/r, r, v, c_x, c_y, c_z, c_Vx, c_Vy, c_Vz, np.sin(r), np.sin(v)])

    #print(f'The shape of data matrix {dmatrix.T.shape}')
    dmatrix = dmatrix.T# + 0.1 * np.random.randn(*dmatrix.T.shape)
    coeff = []
    intercept = []
    #dmatrix -= dmatrix.mean(axis=0)
    #dmatrix /= dmatrix.std(axis=0)

    for target in [p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]:
        #clf = linear_model.Lasso(alpha=1e-3, max_iter=1000)
        clf = linear_model.Ridge()
        #clf = linear_model.HuberRegressor(max_iter=10)
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
