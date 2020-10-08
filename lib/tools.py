import numpy as np

def rk4(f, x, h = 0.01):
    """
    the classical Runge–Kutta method
    """

    k1 = h*f(x)
    k2 = h*f(x+0.5*k1)
    k3 = h*f(x+0.5*k2)
    k4 = h*f(x+k3)
    x = x + (k1+2*k2+2*k3+k4)/6

    return x