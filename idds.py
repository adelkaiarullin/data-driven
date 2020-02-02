import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import progressbar
import random
import datetime
import sindy


def save_submission(arr):
    np.savetxt('submission.csv', arr, delimiter=',', fmt='%d,%f,%f,%f,%f,%f,%f', header='id,x,y,z,Vx,Vy,Vz')


def visualize_trajectory(x, y, z, sx, sy, sz):
    fig = plt.figure()
    print(f'\nLength of trajectory {x.shape}')
    ax = fig.gca(projection='3d')

    ax.plot(x[::2], y[::2], z[::2], lw=0.5, c='r')
    ax.plot(sx[::2], sy[::2], sz[::2], lw=0.5, c='g')

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()


def split_data(values):
    dynamic_data = values[:, -12:] #excract only x y z Vx Vy Vz x_sim y_sim z_sim Vx_sim Vy_sim Vz_sim
    #tunix = np.array([datetime.datetime.strptime(e, "%Y-%m-%dT%H:%M:%S.%f").timestamp() for e in values[:, 1]])
    real_state = dynamic_data[:, :6]
    sim_state = dynamic_data[:, 6:]

    return np.array(real_state, dtype=np.float32), np.array(sim_state, dtype=np.float32)


def generate_in_out_err(state_r, state_s,l=3):
    x, y, z, Vx, Vy, Vz = state_r[:, 0], state_r[:, 1], state_r[:, 2], state_r[:, 3], state_r[:, 4], state_r[:, 5]
    sx, sy, sz, sVx, sVy, sVz =  state_s[:, 0], state_s[:, 1], state_s[:, 2], state_s[:, 3], state_s[:, 4], state_s[:, 5]

    p_x, c_x = x[l:], sx[l:]
    p_y, c_y = y[l:], sy[l:]
    p_z, c_z = z[l:], sz[l:]

    p_Vx, c_Vx = Vx[l:], sVx[l:]
    p_Vy, c_Vy = Vy[l:], sVy[l:]
    p_Vz, c_Vz = Vz[l:], sVz[l:]

    source = np.vstack([c_x, c_y, c_z, c_Vx, c_Vy, c_Vz]).T
    target = np.vstack([p_x, p_y, p_z, p_Vx, p_Vy, p_Vz]).T
    error = target - source

    return source, target, error


def smape(satellite_predicted_values, satellite_true_values): 
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values) 
        /  (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))


def tr_clustering(df):
    sat_id = pd.unique(df['sat_id'])
    data = []
    d = {}
    table = defaultdict(list)
    
    for i in progressbar.progressbar(sat_id): 
        track = np.array(df[df['sat_id'] == i].values[:, -6:], dtype=np.float32)
        d[i] = track
        #mean_vec = track.mean(axis=0)
        mean_std = track.std(axis=0)
        #vec = np.concatenate([mean_vec, mean_std])
        vec =  mean_std
        data.append(vec)

    data = np.vstack(data)

    clustering = SpectralClustering(n_jobs=-1).fit(data)
    print(f'Labels {clustering.labels_}')

    for i, l in zip(sat_id, clustering.labels_):
        table[l].append(d[i])
    
    for k, v in table.items():
        table[k]  = np.concatenate(v, axis=0)

    return clustering, table



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='path to train data.csv')
    parser.add_argument('-test', help='path to test data.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.train, sep=',')
    test_df = pd.read_csv(args.test, sep=',')
    sat_id = pd.unique(test_df['sat_id'])

    hist = []
    d = []
    l = 3
    track = lambda arg: np.vstack([arg[i:arg.shape[0]+i-l] for i in range(l+1)]).T
    dmatrix_creator = lambda arg: np.concatenate([track(e) for e in arg.T], axis=1)

    clustering, table = tr_clustering(test_df)

    for i in progressbar.progressbar(sat_id):
        a, b = split_data(df[df['sat_id'] == i].values)

        loc_var = np.array(test_df[test_df['sat_id'] == i].values[:, -6:], dtype=np.float32)
        label = clustering.predict(loc_var.std(axis=0).reshape(1, 6))
        print(f'Sat id {i}, Cluster {label}')
        context = table[label]

        id_col = np.array(test_df[test_df['sat_id'] == i].values[:, 0], dtype=np.int)
        b = np.concatenate([context, b])
        dmatrix = np.concatenate([b, loc_var], axis=0)
        dmatrix = dmatrix_creator(dmatrix)
        print(f'Dmatrix {dmatrix.shape}')
        dmatrix -= dmatrix.mean(axis=0)
        dmatrix /= dmatrix.std(axis=0)

        source, target, error =  generate_in_out_err(a, b, l)
        clf, h = sindy.test_model(dmatrix[:b.shape[0]-l, :], target, source, error)
        #clf = sindy.train_model(dmatrix[:b.shape[0]-l, :], target, source, error)

        # prediction = clf.predict(dmatrix[b.shape[0]-l:, :]) + loc_var
        # prediction = np.concatenate([id_col.reshape(loc_var.shape[0], 1), prediction], axis=1)
        # d.append(prediction)

        hist.append(h)

        # if i % 100 == 0:
        #     visualize_trajectory(a[:, 0], a[:, 1], a[:, 2], b[:, 0], b[:, 1], b[:, 2])
        #     input('press a key')

    # d = np.concatenate(d, axis=0)
    # save_submission(d)

    hist = np.array(hist)
    print(f'mean {hist.mean()} std {hist.std()}, median {np.median(hist)}')
    plt.hist(hist)
    plt.show()
