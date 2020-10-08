import numpy as np

from lib.ode import Ode

class Lorenz(Ode):
    def __init__(self, params, size):
        super().__init__(params, size)

    def _eq(self, u):
        r, s, b = self.params
        x, y, z = u

        x_dot=s*(y-x)
        y_dot=r*x-y-x*z
        z_dot=x*y-b*z

        return np.array([x_dot, y_dot, z_dot])

    def _jm(self, u):
        r, s, b = self.params
        x_1, x_2, x_3 = u

        jm = np.array([
            [   -s,   s,    0],
            [r-x_3,  -1, -x_1],
            [  x_2, x_1,   -b]
            ])

        return jm