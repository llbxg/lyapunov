from lib.handle import Handle

from lorenz import Lorenz

l = Lorenz(params=(40, 16, 4), size=3)
h = Handle(l)
l_123 = h.lyapunov_exponent_123()
print(l_123)
h.graph()