import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from itertools import islice
from collections import defaultdict
from datetime import datetime
from linear_regression import LinearRegression


startTime = time.time()

"""If files are in the same folder"""
input_path = os.getcwd()+r"/data"
output_path = os.getcwd()
filename = "avocado.csv"
filepath = os.getcwd()+r"/data/{}".format(filename)


def convert_date_type(dates):
    try:
        return datetime.strptime(dates, '%Y-%m-%d')
    except:
        return datetime.strptime(dates, '%d/%m/%Y')


def check_type(val):
    if isinstance(val, float) == True:
        return float(val)
    else:
        return str(val)


def import_data(fname, rowsToRead):
    with open(filepath, 'r') as f:
        reader = csv.reader(f,delimiter=",")
        headers = next(reader)[1:]
        data_dict = defaultdict(list)
        for row in islice(reader, rowsToRead):
            for index, key in enumerate(headers):
                data_dict[key].append(row[index + 1])
    return data_dict


def check_missing_values(col):
    return np.sum(np.isnan(col))


def check_zero(col):
    return np.sum(col == 0.0)


def select_features(d, keys):
	return {x: d[x] for x in d if x not in keys}


def normalize_features(X):
	std = X.std(axis=0)
	std = np.where(std==0, 1, std)
	x_normed = (X - X.mean(axis=0)) /std # avoid division by zero
	return x_normed


# Train val test from scratch
def train_test_split(dummy):
    pass


# k-fold cross validation from scratch
def k_fold_cross_val(dummy):
    pass


# Regularisation from scratch
def regularization(dummy, regu="l1"):
    pass


# PCA from scratch
def get_principal_components(dummy):
    pass



if __name__ == "__main__":
    data = import_data(filepath, 20000)
    headers = data.keys()

    date_v = np.vectorize(convert_date_type)
    int_v = np.vectorize(int)
    float_v = np.vectorize(float)
    str_v = np.vectorize(str)

    # Assign appropriate types to columns
    for key, value in data.items():
        if key == "Date":
            data[key] = date_v(np.array(data[key]))
        elif key == "year":
            data["year"] = int_v(np.array(data[key]))
        elif key == "type":
            data[key] = str_v(np.array(data[key]))
        elif key == "region":
            data[key] = str_v(np.array(data[key]))
        else:
            data[key] = float_v(np.array(data[key]))

    # Check for missing values and zeroes
    for key, value in data.items():
        try:
            print(key, check_missing_values(value), check_zero(value))
        except:
            print(key, "Failed")

    labels = "AveragePrice"
    exclude = ["AveragePrice", "Date", "type", "region", "XLarge Bags", "year"]
    features = [i for i in headers if i not in exclude]
    y = data[labels]
    X = select_features(data, exclude)
    X = np.array([X[i] for i in features]).T

    print(X.shape, y.shape)
    print(features)

    reg = LinearRegression(learning_rate=0.01, n_iters=100)
    X = normalize_features(X)
    costs = reg.fit(X, y)
    weights = reg.weights_
    bias = reg.bias_
    pred = reg.predict(X)

    # Mean absolute error / mean absolute deviation
    print(abs(pred-y).mean())

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax.plot(np.arange(len(costs)), costs)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Sum of squared errors")

    print(features)
    pos = np.arange(len(weights))
    ax2.bar(pos, weights, color='blue',edgecolor='black')
    ax2.set_xticks(pos, features)

    # Need to change this
    #ax2.set_xticklabels(features, rotation=45)

    # normal distribution
    # right skewed, overestimating the price
    ax3.hist(pred-y, bins=20)