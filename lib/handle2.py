from functools import partial
from itertools import accumulate

import numpy as np

from lib.tools import rk4
from lib.handle import Handle

class Handle2(Handle):
    def __init__(self, O, N_ini=0, N_lam=1000, num_ave=100):
        super().__init__(O, N_ini=10, N_lam=N_lam, cal_next=False)

    def lyapunov_exponent_1(self):
        u_tilde = self.u

        lam_1, lambda_1 = [], []
        for n in range(1, self.N_lam+1):
            u_tilde = self.cal_next(u_tilde)
            lam_1.append(np.log(np.abs(self.o._jm(u_tilde))))
            lambda_1.append(sum(lam_1[:n])/(n))

        self.lambda_lists = [lambda_1]
        return  np.average(lambda_1[-self.num_ave:])

    def lyapunov_exponent_123(self):
        return "sorry;;"
