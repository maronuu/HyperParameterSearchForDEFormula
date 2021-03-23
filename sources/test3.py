import os
import sys

import numpy as np
from numpy import power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib

from modules.de_formula import test


### TEST ###

# TEST3
def f3(x):
    return sqrt(1 - x ** 2)

def f3_denom(x):
    return 1

def f3_phi(t):
    return 1 / cosh((pi/2)*sinh(t))

def f3_phi_denom(t):
    return cosh((pi/2)*sinh(t))

def grad_f3_phi_denom(t):
    return (pi/2) * cosh(t) * sinh((pi/2)*sinh(t))

def g3(t):
    return f3_phi(t) * pi * cosh(t) / (1 + cosh(pi * sinh(t)))

xx = np.arange(-1., 1.01, 0.01)
ff_3 = f3(xx)
plt.figure()
plt.plot(xx, ff_3, label="f_3(x)")
plt.xlabel("x")
plt.ylabel("f_3(x)")
plt.title("f_3(x)のグラフ")
plt.savefig("./images/f_3-plot.png")

tt = np.arange(-10, 10, 0.01)
gg_3 = g3(tt)
plt.figure()
plt.plot(tt, gg_3, label="g_3(t)")
plt.xlabel("t")
plt.ylabel("g_3(t)")
plt.title("g_3(t)のグラフ")
plt.savefig("./images/g_3-plot.png")

truth_3 = pi/2
m_list_3 = np.arange(5, 100, 3)
fig_name_3 = "test-3"

test(f3, f3_denom, f3_phi, f3_phi_denom, grad_f3_phi_denom, truth_3, m_list_3, fig_name_3)