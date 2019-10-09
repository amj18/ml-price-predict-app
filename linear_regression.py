import numpy as np


class LinearRegression:
    """
    This is a simple Linear Regression class which allows for the fitting
    of the equation y = wx + b using gradient descent; which minimises the sum
    of squared errors. w is a vector of weights for n number of features
    for the fit, and b is the bias (adjusted error parameter). The model
    evaluates the accuracy using the mean squared error cost function.
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Params
        ____________
        X: array_like, shape = [n_samples, n_features]
        y: array_like, shape = [n_samples, ]

        Returns
        ____________
        self.cost_: array_like
            Array of floats consisting of the cost function over
            number of iterations

        Examples
        ____________
        >>>  lr = LinearRegression(learning_rate=0.01, n_iters=10)
        >>> lr.fit(X_train, y_train)
        >>> costFunction = lr.cost_
        >>> getWeights = lr.weights_
        >>> getBias = lr.bias_
        """
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        self.cost_ = []
        print(X.shape, self.weights_.shape, self.weights_.shape)

        # Implement Gradient descent minimisation
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights_) + self.bias_

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            cost_ = (1/n_samples) * \
                    np.sum([val**2 for val in (y_predicted - y)])
            self.cost_.append(cost_)
        return self.cost_


    def predict(self, X):
        """
        Params
        ____________
        X: array_like, shape = [n_samples, n_features]
            Input features for model

        Returns
        ____________
        y_predicted: array_like, shape = [n_samples, ]
            Predicted values of Linear Regression model

        Examples
        ____________
        >>>  lr = LinearRegression(learning_rate=0.01, n_iters=10)
        >>> lr.fit(X_train, y_train)
        >>> y_pred = lr.predict(X_test)

        """
        y_predicted = np.dot(X, self.weights_) + self.bias_
        return y_predicted