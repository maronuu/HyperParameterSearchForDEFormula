from math import pi, sqrt
import numpy as np
from numpy import cosh, sinh, tanh, exp, log
import matplotlib.pyplot as plt
import japanize_matplotlib


def g(t):
    return (pi/2) * cosh(t) / cosh((pi/2)*sinh(t))


tt = np.arange(-10, 10, 0.01)
gg = g(tt)
plt.plot(tt, gg, label="g(t)")
plt.axhline(y=g(0) * 0.707, c="orange", label="1/√2 peak")
plt.axvline(x=0.696, c="red", label="g(t) = 1/√2 peak を満たすt")
plt.xlabel("t")
plt.ylabel("g(t)")
plt.title("f(x)=1/√(1-x^2)の場合 | g(t) = f(φ(t))φ'(t)のグラフ")
plt.legend()
plt.savefig("./images/sample_g_plot.png")