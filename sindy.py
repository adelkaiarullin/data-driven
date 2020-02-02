import itertools
import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor as gp
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, RationalQuadratic, DotProduct, Sum, Product, Matern
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
import idds
import lorenz


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


def train_model(dmatrix, target, source, error): 
    print(f'The shape of data matrix {dmatrix.shape}')

    clf = MultiOutputRegressor(estimator=KernelRidge(kernel=Product(Product(DotProduct(), Matern()), Product(DotProduct(), Matern()))))# baseline 81.83
    #clf = MultiOutputRegressor(estimator=KernelRidge(kernel=DotProduct()))

    clf.fit(dmatrix, error)

    return clf


def test_model(dmatrix, target, source, error): 
    # print(f'The shape of data matrix {dmatrix.shape}')
    th = int(dmatrix.shape[0] * 0.8)

    clf = MultiOutputRegressor(estimator=KernelRidge(kernel=Product(Product(DotProduct(), Matern()), Product(DotProduct(), Matern()))))# baseline 81.83 on contest
    #clf = MultiOutputRegressor(estimator=gp(kernel=Sum(Product(DotProduct(), Matern()), Product(WhiteKernel(), RationalQuadratic()))))

    step = 1
    #clf.fit(dmatrix, error)
    clf.fit(dmatrix[:th:step, :], error[:th:step, :])
    # score = clf.score(dmatrix[th:, :], error[th:, :])
    predict = clf.predict(dmatrix)
    # print(f'Score {score}')
    
    sm = 100 * (1 - idds.smape(target[th:, :], (predict + source)[th:, :]))
    # print(f'SMAPE {sm}')
    # st_sm = 100 * (1 - idds.smape(target[th:, :], source[th:, :]))
    # print(f'SMAPE {st_sm}')
     

    return clf, sm


if __name__ == '__main__':
    compute_sindy_for_lorenz()
