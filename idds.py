import argparse
import numpy as np
import pandas as pd
from sklearn import linear_model


def split_data(values):
    dynamic_data = values[:, -12:] #excract only x y z Vx Vy Vz x_sim y_sim z_sim Vx_sim Vy_sim Vz_sim
    next_state = dynamic_data[1:, :6]
    curr_state_sim = dynamic_data[:-1, 6:]

    return next_state, curr_state_sim


def smape(satellite_predicted_values, satellite_true_values): 
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values) 
        /  (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='path to train data.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.train, sep=',')

    print(df.columns)
    print(df.head)

    a, b = split_data(df.values)
    del df

    print(f'\nNext states \n{a}')
    print(f'\nCurrent states \n{b}')

    print(f'SMAPE {100 * (1 - smape(a, b))}')

    #clf = linear_model.Lasso(alpha=0.1)
