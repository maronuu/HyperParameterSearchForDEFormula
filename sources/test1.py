import os
import sys

import numpy as np
from numpy import power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib

from modules.de_formula import test


### TEST ###

# TEST1
def f1(x):
    return 1 / sqrt(1-x**2)

def f1_denom(x):
    return sqrt(1-x**2)

def f1_phi(t):
    return cosh((pi/2)*sinh(t))

def f1_phi_denom(t):
    return 1

def grad_f1_phi_denom(t):
    return 0

def g1(t):
    return f1_phi(t) * pi * cosh(t) / (1 + cosh(pi * sinh(t)))

xx = np.arange(-1., 1.01, 0.01)
ff_1 = f1(xx)
plt.figure()
plt.plot(xx, ff_1, label="f_1(x)")
plt.xlabel("x")
plt.ylabel("f_1(x)")
plt.title("f_1(x)のグラフ")
plt.savefig("./images/f_1-plot.png")

tt = np.arange(-10, 10, 0.01)
gg_1 = g1(tt)
plt.figure()
plt.plot(tt, gg_1, label="g_1(t)")
plt.xlabel("t")
plt.ylabel("g_1(t)")
plt.title("g_1(t)のグラフ")
plt.savefig("./images/g_1-plot.png")

truth_1 = pi
m_list_1 = np.arange(5, 100, 3)
fig_name_1 = "test-1"

test(f1, f1_denom, f1_phi, f1_phi_denom, grad_f1_phi_denom, truth_1, m_list_1, fig_name_1)

