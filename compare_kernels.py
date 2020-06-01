import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, ConstantKernel as C
from sim import train_model, generate_data


if __name__ == '__main__':
    dt = .05
    observation = []
    cst_k = C(1.0, (1e-4, 1e3))
    kernels = {'RBF': cst_k * RBF(), 'DotProduct': cst_k * DotProduct(),
               'Matern': cst_k * Matern(), 'RationalQuadratic': cst_k * RationalQuadratic()}

    for k_n, k_f in kernels.items():
        for i in range(1, 31):
            collection = generate_data(i)
            gp, dm, tag = train_model(collection, dt, k_f)
            observation.append(gp.score(dm, tag))

        plt.plot(range(1, len(observation) + 1), observation, lw=2, zorder=9, label=k_n)
        observation.clear()

    plt.xlabel('number of trajectories')
    plt.ylabel('R2 score')
    plt.legend()
    plt.tight_layout()
    plt.show()
