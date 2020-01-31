import itertools
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor as gp
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, RationalQuadratic, DotProduct, CompoundKernel
from sklearn.multioutput import MultiOutputRegressor
# import pandas as pd
# import json
from sklearn.tree import DecisionTreeRegressor
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
    clf.fit(dmatrix.T , (p_z - c_z) / delta)
    print('Coeff')
    print(clf.coef_)
    print('Score')
    print(clf.score(dmatrix.T, (p_z - c_z) / delta))


def compute_sindy(x, y, z, Vx, Vy, Vz, sx, sy, sz, sVx, sVy, sVz):
    l = 5

    (p_x, c_x) = x[l:], sx[l:]#f(x)
    (p_y, c_y) = y[l:], sy[l:]#f(y)
    (p_z, c_z) = z[l:], sz[l:]#f(z)

    (p_Vx, c_Vx) = Vx[l:], sVx[l:]#f(Vx)
    (p_Vy, c_Vy) = Vy[l:], sVy[l:]#f(Vy)
    (p_Vz, c_Vz) = Vz[l:], sVz[l:]#f(Vz)
    
    r = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    v = np.sqrt(sVx ** 2 + sVy ** 2 + sVz ** 2)
    m = (np.square(v) / r)
 
    track = lambda arg: np.vstack([arg[i:arg.shape[0]+i-l] for i in range(l+1)]).T
        
    dmatrix = np.concatenate([track(sx), track(sy), track(sz), track(sVx), track(sVy), track(sVz)], axis=1)
    print(f'The shape of data matrix {dmatrix.shape}')

    dmatrix -= dmatrix.mean(axis=0)
    dmatrix /= dmatrix.std(axis=0)

    th = int(dmatrix.shape[0] * 0.8)

    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    error = target - source

    clf = MultiOutputRegressor(estimator=KernelRidge(kernel=RationalQuadratic()))
    #clf = MultiOutputRegressor(estimator=DecisionTreeRegressor(criterion='mae', max_depth=5))
    
    clf.fit(dmatrix[:th:20, :], error[:th:20, :])
    score = clf.score(dmatrix[th:, :], error[th:, :])
    predict = clf.predict(dmatrix)
    print(f'Score {score}')
    
    sm = 100 * (1 - idds.smape(target, predict + source))
    print(f'SMAPE {sm}')
    st_sm = 100 * (1 - idds.smape(target, source))
    print(f'SMAPE {st_sm}')

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
     
    return sm




if __name__ == '__main__':
    compute_sindy_for_lorenz()
