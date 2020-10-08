from functools import partial
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np

from .tools import rk4

class Handle(object):
    def __init__(self, O, dt=0.01, delta=10**(-3), N_ini=10000, N_lam=10000, num_ave = 100):
        self.o = O
        self.dt, self.size, self.delta = dt, self.o.size, delta
        self.N_ini, self.N_lam, self.num_ave = N_ini, N_lam, num_ave

        self._initial()

        self.lambda_lists = None

    def _initial(self):
        u = np.array([np.random.rand() for _ in range(self.size)])
        for _ in range(self.N_ini):
            u = rk4(self.o, u, h=self.dt)
        self.u = u

    def lyapunov_exponent_1(self):
        u_tilde = self.u

        w = w_0 = np.array([np.random.rand() for _ in range(self.size)]) * self.delta
        z_0 = np.linalg.norm(w_0)

        lambda_1 = []
        for n in range(self.N_lam):
            u_tilde = rk4(self.o, u_tilde)
            f_tilda = partial(self.o.eq_w, u=u_tilde)
            w = rk4(f_tilda, w, h=self.dt)
            lambda_1.append(np.log(np.linalg.norm(w)/z_0)/(n*self.dt))

        self.lambda_lists = lambda_1

        return  np.average(lambda_1[-self.num_ave:])

    def lyapunov_exponent_123(self):
        lambda_lists, lm, w_tau, w_tau_bot = [[[] for _ in range(self.size)] for _ in range(4)]

        u_tilde = self.u

        w_0 = [np.array([np.random.rand() for _ in range(self.size)]) for _ in range(self.size)]
        for i in range(self.size):
            support_gs = 0
            for j in range(0,i,1):
                support_gs += np.dot(w_0[i], w_0[j]) * w_0[j]
            w_0[i] = (v := w_0[i] - support_gs)/np.linalg.norm(v)

        u_hat = [u_tilde + w for w in w_0]

        for _ in range(self.N_lam):
            u_tilde = rk4(self.o, u_tilde, h=self.dt)

            for i in range(self.size):
                u_hat[i] = rk4(self.o, u_hat[i], h=self.dt)
                w_tau[i] = u_hat[i] - u_tilde
                support_gs = 0
                for j in range(0,i,1):
                    support_gs += np.dot(w_tau[i], w_tau[j]) / ((np.linalg.norm(w_tau[j]))**2) * w_tau[j]
                w_tau_bot[i] = w_tau[i] - support_gs

            w_up, w_down = 1, 1
            for i in range(self.size):
                w_up, w_down = np.outer(w_up, w_tau[i]), np.outer(w_down, w_0[i])
                lm[i].append(np.log(np.linalg.norm(w_up) / np.linalg.norm(w_down)))

            for i in range(self.size):
                w_0[i] = np.linalg.norm(w_0[i]) / np.linalg.norm(w_tau_bot[i]) * w_tau_bot[i]
                u_hat[i] = u_tilde + w_0[i]

        for j in range(self.size):
            if j == 0:
                lambda_lists[j] = [ sum(lm[j][:i])/(i*self.dt) for i in range(1, len(lm[j])+1) ]
            else:
                lambda_lists[j] = [ sum(lm[j][:i])/(i*self.dt) - l for i, l in enumerate(lambda_lists[j-1], 1) ]

        self.lambda_lists = lambda_lists

        return [ np.average(l[-self.num_ave:]) for l in lambda_lists]

    def graph(self, xlim=None, ylim=None):
        if self.lambda_lists:
            for i, l in enumerate(self.lambda_lists):
                plt.plot([ t*self.dt for t in range(len(l))],l, label=f"{i}")

        plt.xlim(0, xlim if xlim else self.N_lam*self.dt)
        if ylim:
            plt.ylim(ylim[0],ylim[1])
        plt.xlabel("$t$")
        plt.ylabel("$\lambda_i$")
        plt.legend()
        plt.grid()
        plt.show()