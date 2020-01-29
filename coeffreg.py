import json as js
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    with open('data.json', 'r') as json_file:
        data_dict = js.load(json_file)

    #clf = DecisionTreeRegressor(max_depth=16)
    clf = linear_model.LinearRegression(n_jobs=-1)
    
    data = []
    targets = []
    for k, v in data_dict.items():
        data.append(np.array(v['trajectory']))
        targets += [v['targets']['x'][0]] * data[-1].shape[0]
    
    data = np.vstack(data)
    targets = np.array(targets)
    th = int(data.shape[0] * 0.8)
    print(data.shape, targets.shape)

    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    clf.fit(data[:th, :], targets[:th])
    score = clf.score(data[th:, :], targets[th:])
    print(f'Score {score}')