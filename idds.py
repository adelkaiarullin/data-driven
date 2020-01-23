import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy import stats
import random
import datetime
import sindy


def split_data(values):
    dynamic_data = values[:, -12:] #excract only x y z Vx Vy Vz x_sim y_sim z_sim Vx_sim Vy_sim Vz_sim
    tunix = np.array([datetime.datetime.strptime(e, "%Y-%m-%dT%H:%M:%S.%f").timestamp() for e in values[:, 1]])
    real_state = dynamic_data[:, :6]
    sim_state = dynamic_data[:, 6:]

    return np.array(real_state, dtype=np.float32), np.array(sim_state, dtype=np.float32), tunix


def smape(satellite_predicted_values, satellite_true_values): 
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values) 
        /  (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='path to train data.csv')
    parser.add_argument('-test', help='path to test data.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.train, sep=',')
    #test_df = pd.read_csv(args.test, sep=',')

    #print(df.columns)
    #print(df.head)
    #print(df.describe())
    #print(dt.timestamp())

    hist = []
    coeff = defaultdict(list)

    for i in range(600):
        a, b, t = split_data(df.loc[df['sat_id'] == i].values)
        #print((a - b) / a * 100)
        h, c = sindy.compute_sindy(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:,5], b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:,5], t)
        hist.append(h), coeff['x'].append(c[:, 0]), coeff['y'].append(c[:, 1]), coeff['z'].append(c[:, 2]), coeff['Vx'].append(c[:, 3])
        coeff['Vy'].append(c[:, 4]), coeff['Vz'].append(c[:, 5])

    #a, b = split_data(df.values)
    #score = sindy.compute_sindy(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5])
    #print(f'Score {score}')
    # print(stats.ttest_1samp(np.array(hist), 0.0))
    hist = np.array(hist)
    print(f'mean {hist.mean()} std {hist.std()}, median {np.median(hist)}')
    plt.hist(hist)
    plt.show()


    dynamics = ['1', 't', 'r', '1/r', 'v', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'sin r', 'sin v']
    dynamics = list(itertools.combinations(dynamics, 3)) + ['t', 'r', '1/r', 'v', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'sin r', 'sin v']

    for k, v in coeff.items():
        print(k)
        coeff = np.vstack(v)
        np.set_printoptions(threshold=np.inf)
        print(f'All coeff \n{coeff.shape}')
        for c, d in zip(coeff.T, dynamics):
            (_, p_val) = stats.ttest_1samp(c, 0.0)
            if p_val < 0.05:
                print(p_val,c.shape, d)

    # print(f'SMAPE {100 * (1 - smape(a, b))}')

