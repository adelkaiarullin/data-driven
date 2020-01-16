import pandas as pd
from sklearn import linear_model


if __name__ == '__main__':
    clf = linear_model.Lasso(alpha=0.1)