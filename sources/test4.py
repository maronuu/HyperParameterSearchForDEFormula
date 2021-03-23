import os
import sys

import numpy as np
from numpy import power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib

from modules.de_formula import test


### TEST ###

# TEST4
def f4(x):
    return log(1 + x + 1e-32)

def f4_denom(x):
    return 1

def f4_phi(t):
    # logの中に微小な正の数を足し、zero-divisionを防ぐ。
    return log(1 + tanh((pi/2)*sinh(t))+ 1e-32)

def f4_phi_denom(t):
    return 1

def grad_f4_phi_denom(t):
    return 0

def g4(t):
    return f4_phi(t) * pi * cosh(t) / (1 + cosh(pi * sinh(t)))

xx = np.arange(-1., 1., 0.01)
ff_4 = f4(xx)
plt.figure()
plt.plot(xx, ff_4, label="f_4(x)")
plt.xlabel("x")
plt.ylabel("f_4(x)")
plt.title("f_4(x)のグラフ")
plt.savefig("./images/f_4-plot.png")

tt = np.arange(-10, 10, 0.01)
gg_4 = g4(tt)
plt.figure()
plt.plot(tt, gg_4, label="g_4(t)")
plt.xlabel("t")
plt.ylabel("g_4(t)")
plt.title("g_4(t)のグラフ")
plt.savefig("./images/g_4-plot.png")

truth_4 = 2*log(2) - 2
m_list_4 = np.arange(5, 100, 3)
fig_name_4 = "test-4"

test(f4, f4_denom, f4_phi, f4_phi_denom, grad_f4_phi_denom, truth_4, m_list_4, fig_name_4)