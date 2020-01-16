import argparse
import pandas as pd
from sklearn import linear_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='path to train data.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.train, sep=',')

    print(df.columns)
    print(df.head)
    print(df.values[:, -6:])

    clf = linear_model.Lasso(alpha=0.1)
