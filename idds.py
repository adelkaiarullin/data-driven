import argparse
import numpy as np
import pandas as pd
import sindy


def split_data(values):
    dynamic_data = values[:, -12:] #excract only x y z Vx Vy Vz x_sim y_sim z_sim Vx_sim Vy_sim Vz_sim
    real_state = dynamic_data[:, :6]
    sim_state = dynamic_data[:, 6:]

    return real_state, sim_state


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

    a, b = split_data(df.loc[df['sat_id'] == 0].values)
    del df

    print(f'\nNext states \n{a}')
    print(f'\nCurrent states \n{b}')

    print(f'SMAPE {100 * (1 - smape(a, b))}')

    sindy.compute_sindy(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])
    #sindy.compute_sindy(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5])
