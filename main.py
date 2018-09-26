import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class DGPDE:
    """
    Class implementation for a diagonal Gaussian parametric density estimator that works for d dimensions.
    """

    def __init__(self):
        # TODO
        self.dimension = None
        self.mu = None
        self.covariance = None

    def train(self, train_inputs):
        """
        Method that trains the parameters using a given data set of size Nxd.
        :param train_inputs: Training set having the features as columns and the observations as rows.
        :return: None
        """
        self.dimension = train_inputs.shape[-1]
        self.mu = np.mean(train_inputs, axis=0)
        self.covariance = np.var(train_inputs, axis=0)

    def test(self, test_inputs):
        """
        Function that evaluates the learned parameters on multiple test points.
        :param test_inputs: Test set having the features as columns and the observations as rows.
        :return: the density estimation for each of the test points
        """
        x_minus_mu = test_inputs-self.mu
        exponent = -0.5*np.dot(np.dot((x_minus_mu), np.diag(1/self.covariance)), np.transpose(x_minus_mu))
        normalizing_term = np.power((2*np.pi), (self.dimension/2)) * np.sqrt(np.prod(self.covariance))
        prediction = normalizing_term * np.exp(exponent)
        # return np.log(np.diag(prediction))
        return np.diag(prediction)


class PDE:
    """
    Class implementation for a Parzen density estimator with an isotropic Gaussian kernel that works for d dimensions.
    """

    def __init__(self, variance):
        self.variance = variance
        self.dimension = None
        self.data = None

    def train(self, train_inputs):
        self.dimension = train_inputs.shape[-1]
        transpose_train_inputs = np.transpose(train_inputs)
        self.data = np.reshape(transpose_train_inputs, (1, transpose_train_inputs.shape[0],
                                                        transpose_train_inputs.shape[-1]))

    def test(self, test_inputs):
        tensor_test_inputs = np.repeat(test_inputs[:, :, np.newaxis], self.data.shape[-1], axis=2)
        exponent = -0.5 * np.square(np.linalg.norm(tensor_test_inputs-self.data, axis=1)) / self.variance
        normalizing_term = np.power((2*np.pi), (self.dimension/2)) * np.power(np.sqrt(self.variance), self.dimension)
        gaussian_matrix = normalizing_term * np.exp(exponent)
        prediction = np.sum(gaussian_matrix, axis=1)/test_inputs.shape[0]
        # return np.log(prediction)
        return prediction


def load_dataset(type, feature):
    iris_x, iris_y = datasets.load_iris(True)
    type = iris_x[np.where(iris_y == type)]
    feature = type[:, feature]
    return np.reshape(feature, (feature.size, 1))


def plot_1d(dataset, x_data, y_data, min, max, legend_titles):
    plt.scatter(dataset, np.zeros_like(dataset), label='Dataset')
    for i in range(0, y_data.shape[-1]):
        y = y_data[:, i]
        plt.plot(x_data, y, label=legend_titles[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    N = 100
    dataset = load_dataset(0, 1)
    min = np.min(dataset)
    max = np.max(dataset)
    dx = np.linspace(min, max, N).reshape([-1, 1])
    print('The training data has {} rows and {} columns'.format(dataset.shape[0], dataset.shape[-1]))
    # Generating data to plot DGPDE
    dgpde = DGPDE()
    dgpde.train(dataset)
    dgpde_test = dgpde.test(dx)
    # Generating data to plot PDE with small sigma
    pde = PDE(0.1)
    pde.train(dataset)
    pde_test_small = pde.test(dx)
    # Generating data to plot PDE with large sigma
    pde = PDE(10)
    pde.train(dataset)
    pde_test_large = pde.test(dx)
    # Generating data to plot PDE with optimal sigma
    pde = PDE(1)
    pde.train(dataset)
    pde_test_optimal = pde.test(dx)
    # Stack generated data
    stack = np.stack((dgpde_test, pde_test_small, pde_test_large, pde_test_optimal), axis=1)
    # Plot data
    plot_1d(dataset, dx, stack, 0, 0, ['A', 'B', 'C', 'D'])