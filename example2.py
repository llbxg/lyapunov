from lib.handle2 import Handle2

from logistic import Logistic

l = Logistic(params=(4), size=1)
h = Handle2(l)
lam1 = h.lyapunov_exponent_1()
print(lam1)
h.graph()