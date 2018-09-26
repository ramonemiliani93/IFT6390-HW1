import numpy as np
import matplotlib.pyplot as plt


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
        print('The empirical mean of the training data is: {}'.format(self.mu))
        self.covariance = np.var(train_inputs, axis=0)
        print('The empirical diagonal covariance matrix of the training data is: {}'.format(self.covariance))

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
        return prediction

    def plot(self):
        # TODO
        pass


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
        prediction = np.sum(gaussian_matrix, axis=1)/gaussian_matrix.shape[-1]
        return prediction


    def plot(self):
        # TODO
        pass


if __name__ == "__main__":
    mean = [0, 0, 5]
    cov = [[1, 0, 0], [0, 100, 0], [0, 0, 5]]
    t_inputs = [[1, 2, 3], [4, 5, 6], [6, 7, 8], [9, 10, 11]]
    x = np.random.multivariate_normal(mean, cov, 5000)
    print('The training data has {} rows and {} columns'.format(x.shape[0], x.shape[-1]))
    dgpde = DGPDE()
    dgpde.train(x)
    dgpde.test(t_inputs)
    pde = PDE(2)
    pde.train(x)
    pde.test(np.array(t_inputs))
