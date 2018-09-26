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
        exponent = -0.5 * np.square(np.linalg.norm(tensor_test_inputs-self.data, axis=1, ord=None)) / self.variance
        normalizing_term = 1/((2*np.pi)**(self.dimension/2) * (np.sqrt(self.variance)**(self.dimension)))
        gaussian_matrix = normalizing_term * np.exp(exponent)
        prediction = np.sum(gaussian_matrix/self.data.shape[-1], axis=1)
        # return np.log(prediction)
        return prediction


def load_dataset(type, feature):
    iris_x, iris_y = datasets.load_iris(True)
    type = iris_x[np.where(iris_y == type)]
    feature = type[:, feature]
    return np.reshape(feature, (feature.size, 1))


def plot(dataset, results, dimension):
    if dimension == 1:
        plt.scatter(dataset, np.zeros_like(dataset), label='Dataset')
        for x, y, label in results:
            plt.plot(x, y, label=label)
        plt.legend()
        plt.show()
    elif dimension == 2:
        plt.scatter(dataset[:, 0], dataset[:, 1], label='Dataset')
        for x, y, label in results:
            plt.contour(x[0], x[1], y, label=label)
        plt.legend()
        plt.show()
    else:
        print("Can't plot data in that dimension")


if __name__ == "__main__":
    # ONE DIMENSION EXERCISE
    N = 100
    dataset = load_dataset(0, 1)
    min = np.min(dataset)
    max = np.max(dataset)
    dx = np.linspace(min, max, N).reshape([-1, 1])
    results = []
    print('The training data has {} rows and {} columns'.format(dataset.shape[0], dataset.shape[-1]))
    # Generating data to plot DGPDE
    dgpde = DGPDE()
    dgpde.train(dataset)
    dgpde_y = dgpde.test(dx)
    results.append((dx, dgpde_y, 'GPDE'))
    # Generating data to plot PDE with small sigma
    pde = PDE(0.001)
    pde.train(dataset)
    pde_y_small = pde.test(dx)
    results.append((dx, pde_y_small, 'PDE $\sigma^2=0.001$'))
    # Generating data to plot PDE with large sigma
    pde = PDE(1)
    pde.train(dataset)
    pde_y_large = pde.test(dx)
    results.append((dx, pde_y_large, 'PDE $\sigma^2=1.000$'))
    # Generating data to plot PDE with optimal sigma
    optimal_sigma = 1.06*np.sqrt(np.var(dataset))*dataset.shape[0]**(-1/5) # Silverman's rule of thumb
    pde = PDE(optimal_sigma)
    pde.train(dataset)
    pde_y_optimal = pde.test(dx)
    results.append((dx, pde_y_optimal, 'PDE $\sigma^2={0:.3f}$'.format(optimal_sigma)))
    # Plot data
    plot(dataset, results, 1)

    # TWO DIMENSION EXERCISE
    N = 100
    x1 = load_dataset(0, 1)
    x2 = load_dataset(0, 2)
    min_x1 = np.min(x1)
    max_x1 = np.max(x1)
    min_x2 = np.min(x2)
    max_x2 = np.max(x2)
    dataset = np.hstack((x1, x2))
    print(dataset.shape)
    dx1 = np.linspace(min_x1, max_x1, N).reshape([-1, 1])
    dx2 = np.linspace(min_x2, max_x2, N).reshape([-1, 1])
    grid = np.meshgrid(dx1, dx2)
    dx = np.hstack((grid[0].reshape([-1, 1]), grid[1].reshape([-1, 1])))
    results = []
    print('The training data has {} rows and {} columns'.format(dataset.shape[0], dataset.shape[-1]))
    # Generating data to plot DGPDE
    dgpde = DGPDE()
    dgpde.train(dataset)
    dgpde_y = dgpde.test(dx)
    print(dgpde_y.shape)
    results.append((grid, dgpde_y.reshape(grid[0].shape), 'GPDE'))
    # Generating data to plot PDE with small sigma
    pde = PDE(0.001)
    pde.train(dataset)
    pde_y_small = pde.test(dx)
    results.append((grid, pde_y_small.reshape(grid[0].shape), 'PDE $\sigma^2=0.001$'))
    # Generating data to plot PDE with large sigma
    pde = PDE(1)
    pde.train(dataset)
    pde_y_large = pde.test(dx)
    results.append((grid, pde_y_large.reshape(grid[0].shape), 'PDE $\sigma^2=1.000$'))
    # Generating data to plot PDE with optimal sigma
    optimal_sigma = 1.06 * np.sqrt(np.var(dataset)) * dataset.shape[0] ** (-1 / 5)  # Silverman's rule of thumb
    pde = PDE(optimal_sigma)
    pde.train(dataset)
    pde_y_optimal = pde.test(dx)
    results.append((grid, pde_y_optimal.reshape(grid[0].shape), 'PDE $\sigma^2={0:.3f}$'.format(optimal_sigma)))
    # Plot data
    plot(dataset, results, 2)