import numpy as np


# from numpy import dot, random, exp

class NeuralNetwork:
    def __init__(self):
        # set seed
        np.random.seed(1)
        # generate initial weights matrix [ 3x1 ]
        self.neural_network_weights = np.random.random((3, 1)) * 2 - 1
        print 'neural network weights:\n', self.neural_network_weights

    def train(self, X, y, iteration_num):
        for i in xrange(iteration_num):
            self.__back_propagation(X, y)

    def predict(self, X):
        return self.__sigmod(np.dot(X, self.neural_network_weights))

    def __back_propagation(self, X, y):
        """
        X is [ nx3 ] training input
        """
        # print 'X:\n', X
        # calculate result
        output = self.__sigmod(np.dot(X, self.neural_network_weights))
        # calculate error
        error = output - y
        # print 'output, y, error:\n', output, y, error
        print 'output:\n', output
        # descent gradient
        adjustment = np.dot(X.T, self.__sigmod_slope_calculate(output) * error)
        # adjust the weight
        self.neural_network_weights += adjustment

    def __sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmod_slope_calculate(self, x):
        return x * (1 - x)


def main():
    nn = NeuralNetwork()

    # generate training data:
    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    nn.train(X, y, 10000)

    print 'predict result: ', nn.predict([1, 0, 0])
    return


if __name__ == '__main__':
    main()
