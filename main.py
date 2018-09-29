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
        print(self.covariance)

    def test(self, test_inputs, log_density):
        """
        Function that evaluates the learned parameters on multiple test points.
        :param test_inputs: Test set having the features as columns and the observations as rows.
        :param log_density: Boolean, if True returns log(density).
        :return: prediction: the density or log-density estimation for each of the test points.
        """
        x_minus_mu = test_inputs-self.mu
        exponent = -0.5*np.sum(np.multiply(np.dot((x_minus_mu), np.diag(1/self.covariance)).T, np.transpose(x_minus_mu)), axis=0)
        normalizing_term = 1/((2*np.pi)**(self.dimension/2) * np.sqrt(np.prod(self.covariance)))
        prediction = normalizing_term * np.exp(exponent)
        if log_density:
            prediction = np.log(prediction)
        return prediction

class PDE:
    """
    Class implementation for a Parzen density estimator with an isotropic Gaussian kernel that works for d dimensions.
    """

    def __init__(self, variance):
        self.variance = variance
        self.dimension = None
        self.data = None

    def train(self, train_inputs):
        """
        Method that trains the parameters using a given data set of size Nxd.
        :param train_inputs: Training set having the features as columns and the observations as rows.
        :return: None
        """
        self.dimension = train_inputs.shape[-1]
        transpose_train_inputs = np.transpose(train_inputs)
        self.data = np.reshape(transpose_train_inputs, (1, transpose_train_inputs.shape[0],
                                                        transpose_train_inputs.shape[-1]))

    def test(self, test_inputs, log_density=True):
        """
        Function that evaluates the learned parameters on multiple test points.
        :param test_inputs: Test set having the features as columns and the observations as rows.
        :param log_density: Boolean, if True returns log(density).
        :return: prediction: the density or log-density estimation for each of the test points.
        """
        tensor_test_inputs = np.repeat(test_inputs[:, :, np.newaxis], self.data.shape[-1], axis=2)
        exponent = -0.5 * np.square(np.linalg.norm(tensor_test_inputs-self.data, axis=1, ord=None)) / self.variance
        normalizing_term = 1/((2*np.pi)**(self.dimension/2) * (np.sqrt(self.variance)**self.dimension))
        gaussian_matrix = normalizing_term * np.exp(exponent)
        prediction = np.sum(gaussian_matrix/self.data.shape[-1], axis=1)
        if log_density:
            prediction = np.log(prediction)
        return prediction


def load_dataset(type, feature):
    """
    Helper function to retrieve a specific class and feature from the iris dataset.
    :param type: Class to be selected
    :param feature: feature to be selected
    :return: ds: Column vector with corresponding values
    """
    iris_x, iris_y = datasets.load_iris(True)
    type = iris_x[np.where(iris_y == type)]
    feature = type[:, feature]
    ds = np.reshape(feature, (feature.size, 1))
    return ds


def plot(dataset, results, xlabel, ylabel, title, save, name=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(dataset, np.zeros_like(dataset), label='Dataset', color='black')
        for x, y, label, color in results:
            plt.plot(x, y, label=label, color=color)
        plt.legend()
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.margins(y=0)
        plt.show()
        if save and name !=  None:
            fig.savefig(name+'.png', bbox_inches='tight')


def contour(dataset, results, save, name=None):
        i = 1
        for x, y, label in results:
            fig = plt.figure()
            plt.scatter(dataset[:, 0], dataset[:, 1], label='Dataset', color='black')
            cs = plt.contour(x[0], x[1], y, cmap='inferno')
            plt.clabel(cs, inline=1, fontsize=8)
            plt.xlabel(label[0])
            plt.ylabel(label[1])
            plt.title(label[2])
            plt.margins(x=0)
            plt.margins(y=0)
            plt.grid()
            plt.show()
            if save and name != None:
                fig.savefig(name+'-{}.png'.format(i), bbox_inches='tight')
            i += 1


if __name__ == "__main__":
    # ONE DIMENSION EXERCISE
    N = 100
    dataset = load_dataset(0, 3) # Setosa - Petal Width
    min = np.min(dataset)
    max = np.max(dataset)
    dx = np.linspace(min, max, N).reshape([-1, 1])
    results = []
    print('The training data has {} rows and {} columns'.format(dataset.shape[0], dataset.shape[-1]))
    # Generating data to plot DGPDE
    dgpde = DGPDE()
    dgpde.train(dataset)
    dgpde_y = dgpde.test(dx, False)
    results.append((dx, dgpde_y, 'GPDE', 'blue'))
    # Generating data to plot PDE with small sigma
    pde = PDE(0.0001)
    pde.train(dataset)
    pde_y_small = pde.test(dx, False)
    results.append((dx, pde_y_small, 'PDE $\sigma^2=0.001$', 'red'))
    # Generating data to plot PDE with large sigma
    pde = PDE(1)
    pde.train(dataset)
    pde_y_large = pde.test(dx, False)
    results.append((dx, pde_y_large, 'PDE $\sigma^2=1.000$', 'magenta'))
    # Generating data to plot PDE with optimal sigma
    optimal_variance = (1.06*np.sqrt(np.var(dataset))*dataset.shape[0]**(-1/5))**2 # Silverman's rule of thumb
    pde = PDE(optimal_variance)
    pde.train(dataset)
    pde_y_optimal = pde.test(dx, False)
    results.append((dx, pde_y_optimal, 'PDE $\sigma^2={0:.3f}$'.format(optimal_variance), 'green'))
    # Plot data
    xlabel = 'Petal width for Setosa [cm]'
    ylabel = 'Density'
    title = 'Density plot for petal width of Setosa'
    plot(dataset, results, xlabel, ylabel, title, True, name='1d-density')

    # TWO DIMENSION EXERCISE
    N = 100
    x1 = load_dataset(0, 0)  # Setosa - Sepal Length
    x2 = load_dataset(0, 2)  # Setosa - Petal Length
    min_x1 = np.min(x1)
    max_x1 = np.max(x1)
    min_x2 = np.min(x2)
    max_x2 = np.max(x2)
    dataset = np.hstack((x1, x2))
    dx1 = np.linspace(min_x1, max_x1, N).reshape([-1, 1])
    dx2 = np.linspace(min_x2, max_x2, N).reshape([-1, 1])
    grid = np.meshgrid(dx1, dx2)
    dx = np.hstack((grid[0].reshape([-1, 1]), grid[1].reshape([-1, 1])))
    results = []
    print('The training data has {} rows and {} columns'.format(dataset.shape[0], dataset.shape[-1]))
    # Shared plot information
    xlabel = 'Sepal length for Setosa [cm]'
    ylabel = 'Petal length for Setosa [cm]'
    # Generating data to plot DGPDE
    dgpde = DGPDE()
    dgpde.train(dataset)
    dgpde_y = dgpde.test(dx, False)
    label = (xlabel, ylabel, 'Diagonal Gaussian parametric estimator')
    results.append((grid, dgpde_y.reshape(grid[0].shape), label))
    # Generating data to plot PDE with small sigma
    pde = PDE(0.001)
    pde.train(dataset)
    pde_y_small = pde.test(dx, False)
    label = label = (xlabel, ylabel, 'Parzen estimator with $\sigma^2=0.001$')
    results.append((grid, pde_y_small.reshape(grid[0].shape), label))
    # Generating data to plot PDE with large sigma
    pde = PDE(5)
    pde.train(dataset)
    pde_y_large = pde.test(dx, False)
    label = label = (xlabel, ylabel, 'Parzen estimator with $\sigma^2=5.000$')
    results.append((grid, pde_y_large.reshape(grid[0].shape), label))
    # Generating data to plot PDE with optimal sigma
    # Scott's rule
    optimal_variance = (np.mean(np.sqrt(np.var(dataset, axis=0)) * dataset.shape[0] ** (-1 / (4+dataset.shape[1]))))**2
    pde = PDE(optimal_variance)
    pde.train(dataset)
    pde_y_optimal = pde.test(dx, False)
    label = label = (xlabel, ylabel, 'Parzen estimator with $\sigma^2={0:.3f}$'.format(optimal_variance))
    results.append((grid, pde_y_optimal.reshape(grid[0].shape), label))
    # Plot data
    contour(dataset, results, True, name='2d-density')
