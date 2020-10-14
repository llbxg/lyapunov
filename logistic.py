import numpy as np

from lib.ode import Ode

class Logistic(Ode):
    def __init__(self, params, size):
        super().__init__(params, size)

    def _eq(self, u):
        a = self.params
        return np.array([a*u[0]*(1-u[0])])

    def _jm(self, u):
        a = self.params
        return np.array([-2*a*u[0]+a])