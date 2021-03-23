import os
import sys

import numpy as np
from numpy import e, power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib

from modules.de_formula import test


### TEST ###

# TEST4
def f5(x):
    return exp(-x)*sin(100*pi*x)

def f5_denom(x):
    return 1

def f5_phi(t):
    return exp(-(tanh((pi/2)*sinh(t)))) * sin(100*pi*tanh((pi/2)*sinh(t)))

def f5_phi_denom(t):
    return 1

def grad_f5_phi_denom(t):
    return 0

def g5(t):
    return f5_phi(t) * pi * cosh(t) / (1 + cosh(pi * sinh(t)))

xx = np.arange(-1., 1., 0.0001)
ff_5 = f5(xx)
plt.figure()
plt.plot(xx, ff_5, label="f_5(x)")
plt.xlabel("x")
plt.ylabel("f_5(x)")
plt.title("f_5(x)のグラフ")
plt.savefig("./images/f_5-plot.png")

tt = np.arange(-10, 10, 0.0001)
gg_5 = g5(tt)
plt.figure()
plt.plot(tt, gg_5, label="g_5(t)")
plt.xlabel("t")
plt.ylabel("g_5(t)")
plt.title("g_5(t)のグラフ")
plt.savefig("./images/g_5-plot.png")

truth_5 = 100*(e**2-1)*pi/(e+e*(100*pi)**2)
m_list_5 = np.arange(5, 100, 3)
fig_name_5 = "test-5"

test(f5, f5_denom, f5_phi, f5_phi_denom, grad_f5_phi_denom, truth_5, m_list_5, fig_name_5)