import numpy as np


def polynomial_weights(poly):
    poly_class = {
        'beta': BetaWeights(1., 5.),
        'beta_nz': BetaWeights(1., 5.),
        'expalmon': ExpAlmonWeights(-1., 0.)
    }

    return poly_class[poly]


class WeightMethod(object):
    def __init__(self):
        pass

    def weights(self):
        pass


class BetaWeights(WeightMethod):
    def __init__(self, theta1, theta2, theta3=None):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def weights(self, nlags):
        """ Evenly-spaced beta weights
        """
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)

        beta_vals = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)

        beta_vals = beta_vals / sum(beta_vals)

        if self.theta3 is not None:
            w = beta_vals + self.theta3
            return w / sum(w)

        return beta_vals

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @property
    def num_params(self):
        return 2 if self.theta3 is None else 3

    @staticmethod
    def init_params():
        return np.array([1., 5.])


class ExpAlmonWeights(WeightMethod):
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def weights(self, nlags):
        """
        Exponential Almon weights

        Returns:
            array: Array of weights

        """
        ilag = np.arange(1, nlags + 1)
        z = np.exp(self.theta1 * ilag + self.theta2 * ilag ** 2)
        return z / sum(z)

    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params

        w = self.weights(x.shape[1])

        return np.dot(x, w), np.tile(w.T, (x.shape[1], 1))

    @property
    def num_params(self):
        return 2

    @staticmethod
    def init_params():
        return np.array([-1., 0.])
