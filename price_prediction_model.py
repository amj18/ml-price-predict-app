import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from itertools import islice
from collections import defaultdict
from datetime import datetime
from linear_regression import LinearRegression
import random
from scipy.stats import norm

startTime = time.time()

"""If files are in the same folder"""
input_path = os.getcwd()+r"/data"
output_path = os.getcwd()
filename = "avocado.csv"
filepath = os.getcwd()+r"/data/{}".format(filename)


def convert_date_type(dates):
    """
    Convert string dates to datetime objects.
    """
    try:
        return datetime.strptime(dates, '%Y-%m-%d')
    except ValueError:
        return datetime.strptime(dates, '%d/%m/%Y')


def check_type(var):
    """
    Check whether variable is type float or string and return its
    correct type.
    """
    if isinstance(var, float):
        return float(var)
    else:
        return str(var)


def import_data(fname, rowsToRead):
    """
    Imports data from csv file and appends to a data dictionary with unique
    key based on the column name, identified by the header.

    Parameters
    ____________
    fname: str
        File path and name is in string format
    rowsToRead: int
        Read file in chunks row by row

    Returns
    ____________
    data_dict: dictionary object
        Dictionary of lists with unique key based on the header
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        headers = next(reader)[1:]
        data_dict = defaultdict(list)
        for row in islice(reader, rowsToRead):
            for index, key in enumerate(headers):
                data_dict[key].append(row[index + 1])
    return data_dict


def check_missing_values(col):
    """
    Checks for missing values in an array.
    """
    return np.sum(np.isnan(col))


def check_zero(col):
    """
    Checks for zeroes or empty values in an array
    """
    return np.sum(col == 0.0)


def run_missing_value_check():
    """
    Checks for missing values and zeroes in all the columns of the dictionary
    object.
    """
    print("\n### CHECKING FOR MISSING VALUES AND ZEROES ###")
    for key, value in data.items():
        try:
            print(key, check_missing_values(value), check_zero(value))
        except TypeError:
            print(key, "Failed")
    print("### END ###\n")


def bootstrap_sample(data):
    """
    Randomly sample elements of the original dataset.
    """
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data, stats_fn, num_samples):
    """
    Evaluate minimisation of weights in the regression model on a selected
    number of samples of the bootstrapped data.
    """
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


def estimate_sample_beta(sample):
    """
    Estimate the weights (betas) of the sampled data using gradient descent.
    """
    x_s, y_s = zip(*sample)
    reg.fit(x_s, y_s)
    betas = reg.weights_
    return betas


def p_value(beta_hat_j, sigma_hat_j):
    """
    Estimate the p-values for the features of the regression model (the
    betas) assuming a Normal distribution.

    Parameters
    ____________
    beta_hat_j: float or array-like
        Float or vector of weights for the regression model
    sigma_hat_j: float or array-like
        These are the standard errors for the weights

    Returns
    ____________
    p-value: float or array-like
        Float or vector of p-values for the weights (the features)
    """
    if beta_hat_j > 0:
        return 2 - (1 * norm.cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * norm.cdf(beta_hat_j / sigma_hat_j)


def dimension_check():
    """
    Checks for the dimensions of the matrices and vectors.
    """
    print("### DIMENSION CHECK ###")
    print(X.shape,
          y.shape,
          X_train.shape,
          y_train.shape,
          X_test.shape,
          y_test.shape,
          weights.shape)
    print("### END ###")


def select_features(d, keys):
    """
    Select relevant features and exclude non-relevant ones.

    Parameters
    ____________
    d: dictionary of arrays
    keys: array-like
        Keys in array must be strings

    Returns
    ____________
    x: dictionary
        dict object consisting of arrays of the relevant features
    """
    return {x: d[x] for x in d if x not in keys}


def normalize_features(X):
    """
    Mean normalisation / Standardisation, (naively) assumes data follows a
    Normal distribution.

    Parameters
    ____________
    X: array-like

    Returns
    ____________
    x_normed: array-like (numpy)
    """
    std = X.std(axis=0)
    std = np.where(std == 0, 1, std)  # to avoid division by zero
    x_normed = (X - X.mean(axis=0)) / std
    return x_normed


def split_data(data, prob):
    """
    Split data into fractions determined by prob and (1-prob).

    Parameters
    ____________
    data: array-like
    prob: float

    Returns
    ____________
    results: array-like
    """
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(x, y, test_pct):
    """
    Split data into train and test sets.

    Parameters
    ____________
    x: array-like
    y: array-like
    test_pct: float

    Returns
    ____________
    x_train, y_train, x_test, y_test: tuples consisting of array-like
                                      objects
    """
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(42)  # Setting random seed as random will be called later.
    rows_to_read = 20000
    data = import_data(filepath, rows_to_read)
    headers = data.keys()

    # Vectorize functions to allow application of functions to large arrays
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
    run_missing_value_check()

    labels = "AveragePrice"
    exclude = ["AveragePrice",
               "Date",
               "type",
               "region",
               "XLarge Bags",
               "year"]
    features = [i for i in headers if i not in exclude]
    num_columns = np.arange(0, len(features), 1)
    y = data[labels]
    X = select_features(data, exclude)
    X = np.array([X[i] for i in features]).T
    X = normalize_features(X)

    # FItting model
    reg = LinearRegression(learning_rate=0.01, n_iters=300)

    # Train, test split with 80/20 ratio
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.20)

    # Quick hack to get around tuple to array conversion
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    costs = reg.fit(X_train, y_train)
    weights = reg.weights_
    bias = reg.bias_
    pred_train = reg.predict(X_train)
    pred = reg.predict(X_test)

    dimension_check()

    # Mean absolute error / mean absolute deviation of training set
    print("""\n### Mean absolute error / mean absolute deviation
          of training set ###""")
    print(abs(pred_train-y_train).mean())

    # Mean absolute error / mean absolute deviation of test set
    print("""\n### Mean absolute error / mean absolute deviation
          of test set ###""")
    print(abs(pred-y_test).mean())

    bootstrap_betas = bootstrap_statistic(list(zip(X_train, y_train)),
                                          estimate_sample_beta, 20)
    bootstrap_standard_errors = [np.std([beta[i] for beta in bootstrap_betas])
                                 for i in range(len(features))]

    # Calculate p-values for 7 features
    b1 = p_value(weights[0], bootstrap_standard_errors[0])
    b2 = p_value(weights[1], bootstrap_standard_errors[1])
    b3 = p_value(weights[2], bootstrap_standard_errors[2])
    b4 = p_value(weights[3], bootstrap_standard_errors[3])
    b5 = p_value(weights[4], bootstrap_standard_errors[4])
    b6 = p_value(weights[5], bootstrap_standard_errors[5])
    b7 = p_value(weights[6], bootstrap_standard_errors[6])

    print("\nP-values for the features using bootstrapping")
    print(features)
    print(weights, bootstrap_standard_errors)
    print(b1, b2, b3, b4, b5, b6, b7)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax.plot(np.arange(len(costs)), costs)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean squared error (MSE)")

    # Box plot for the features based on their fitted weights
    N = len(features)
    pos = np.arange(N)
    margin = 1
    width = (1.-2.*margin)/N
    rects1 = ax2.bar(pos, weights, color='w', ls="-", lw=1, edgecolor="b")
    ax2.set_xticks(pos + width / 2)
    ax2.set_xticklabels(features, fontsize=10, rotation=90)

    # Skewed Normal distribution
    # Negative skewed, meaning model is overestimating the price on average
    ax3.hist(pred-y_test, bins=20)
    ax3.set_xlabel("(Predicted - y_test)")
    ax3.set_ylabel("Counts")

    endTime = time.time()
    scriptRunTime = endTime - startTime
    print("\n### Total runtime = {:0.2f} s. ###\n".format(scriptRunTime))

    # Comments:
    # The feature: Total Volume, 4046, 4770 and Large Bags "appear" to be
    # statistically significant in predicting the price of avocadoes
    # due to the zero p-values.
