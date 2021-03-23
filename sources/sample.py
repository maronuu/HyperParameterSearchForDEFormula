
import numpy as np
from numpy import log, exp, sinh
import matplotlib.pyplot as plt
import japanize_matplotlib


tt = np.arange(0, 10.5835, 0.001)

def calc_1st_approximation():
        tt = np.arange(0, 10.5835, 0.001)
        alpha_list = np.arange(2.0-0.0002, 6.0, 0.01)
        t_alpha = []
        for t in tt:
            decided_alpha = 0
            for alpha in alpha_list:
                if sinh(t) > exp(t) / alpha:
                    decided_alpha = alpha
                    break
            t_alpha.append(alpha)
        return t_alpha

def calc_2nd_approximation():
    tt = np.arange(0, 10.5835, 0.001)
    gamma_list = np.arange(0.01, 0.5, 0.001)
    t_gamma = []
    for t in tt:
        decided_gamma = 0
        for gamma in gamma_list:
            if exp(t) < gamma * exp(exp(t)):
                decided_gamma = gamma
                break
        t_gamma.append(gamma)
    return t_gamma

t_alpha = calc_1st_approximation()
t_gamma = calc_2nd_approximation()


plt.figure()
plt.plot(tt, t_alpha, label="α_min(t)")
plt.legend()
plt.xlabel("t")
plt.ylabel("α_min(t)")
plt.title("α_min(t)のグラフ")
plt.savefig("./images/alpha-test.png")


plt.figure()
plt.plot(tt, t_gamma, label="γ_min(t)")
plt.legend()
plt.xlabel("t")
plt.ylabel("γ_min(t)")
plt.title("γ_min(t)のグラフ")
plt.savefig("./images/gamma-test.png")