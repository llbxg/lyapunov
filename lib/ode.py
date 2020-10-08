import numpy as np

class Ode(object):
    def __init__(self, params, size):
        self.params = params
        self.size = size

    def __call__(self, u):
        return self._eq(u)

    def _eq(self, u):
        return u

    def _jm(self, u):  # Jacobian Matrix
        return np.ones((s:=u.size) * s).reshape((s,s))

    def eq_w(self, w, u):
        jm = self._jm(u)
        return np.dot(jm, w)