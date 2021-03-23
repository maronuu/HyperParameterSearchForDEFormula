import os
import sys

import numpy as np
from numpy import power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib

from modules.de_formula import test


### TEST ###

# TEST2
def f2(x):
    return 1 / (1 + x **2)

def f2_denom(x):
    return 1 + x ** 2

def f2_phi(t):
    return 1 / (1 + tanh((pi/2)*sinh(t)) ** 2)

def f2_phi_denom(t):
    return 1 + tanh((pi/2)*sinh(t)) ** 2

def grad_f2_phi_denom(t):
    return pi * tanh((pi/2)*sinh(t)) * cosh(t) / (cosh((pi/2)*sinh(t)))**2

def g2(t):
    return f2_phi(t) * pi * cosh(t) / (1 + cosh(pi * sinh(t)))

xx = np.arange(-1., 1.01, 0.01)
ff_2 = f2(xx)
plt.figure()
plt.plot(xx, ff_2, label="f_2(x)")
plt.xlabel("x")
plt.ylabel("f_2(x)")
plt.title("f_2(x)のグラフ")
plt.savefig("./images/f_2-plot.png")

tt = np.arange(-10, 10, 0.01)
gg_2 = g2(tt)
plt.figure()
plt.plot(tt, gg_2, label="g_2(t)")
plt.xlabel("t")
plt.ylabel("g_2(t)")
plt.title("g_2(t)のグラフ")
plt.savefig("./images/g_2-plot.png")


truth_2 = pi / 2.
m_list_2 = np.arange(5, 100, 3)
fig_name_2 = "test-2"

test(f2, f2_denom, f2_phi, f2_phi_denom, grad_f2_phi_denom, truth_2, m_list_2, fig_name_2)