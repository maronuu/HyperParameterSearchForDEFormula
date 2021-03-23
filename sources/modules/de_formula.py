import os
import sys

import numpy as np
from numpy import power, sinh, cosh, tanh, pi, log, sqrt, exp, cos, sin
import matplotlib.pyplot as plt
import japanize_matplotlib



class DEFormula:
    """二重指数関数型数値積分公式(DE公式)
    [-1, 1]の定積分を行う。
    変数変換は x = φ(t) = tanh((pi/2)*sinh(t))
    """

    def __init__(self, f, f_denom, f_phi, f_phi_denom, grad_f_phi_denom, truth):
        """ クラスの初期化
        Arguments
        ---------
            f(float -> float): 被積分関数
            f_denom(float->float): fの分母
            f_phi(float -> float): 合成関数f(φ(t))
            f_phi_denom(float -> float): f_phiの分母
            grad_f_phi_denom(float -> float): d/dt(f_phi_denom)
        Parameters
        ----------
            BETA(float): パラメータ。打ち切りmから幅hを計算する際に用いる。
            EPS(float): 特異性を判定する際に用いる閾値。非常に小さい数。
        """
        self.f = f
        self.f_denom = f_denom
        self.f_phi = f_phi
        self.f_phi_denom = f_phi_denom
        self.grad_f_phi_denom = grad_f_phi_denom
        self.beta = None
        self.EPS = 1e-10
        self.TRUTH = truth
        
        self.d = pi/2

    def phi(self, t):
        """変数変換 x = φ(t)
        """
        return tanh((pi/2)*sinh(t))

    def grad_phi(self, t):
        """ d/dt φ(t)
        """
        return pi * cosh(t) / (1 + cosh(pi * sinh(t)))
    
    def g(self, t):
        """ g(t) := f(φ(t)) * d/dt φ(t)
        """
        return self.f_phi(t) * self.grad_phi(t)
    
    def calc_1st_approximation(self):
        tt = np.arange(10.5835, -0.001, -0.001)
        alpha_list = np.arange(2.0-0.02, 6.0, 0.01)
        t_alpha = []
        for t in tt:
            decided_alpha = 0
            for alpha in alpha_list:
                if sinh(t) > exp(t) / alpha:
                    decided_alpha = alpha
                    break
            t_alpha.append(decided_alpha)
        return t_alpha

    def calc_2nd_approximation(self):
        tt = np.arange(10.5835, -0.001, -0.001)
        gamma_list = np.arange(0.01, 0.5, 0.001)
        t_gamma = []
        for t in tt:
            decided_gamma = 0
            for gamma in gamma_list:
                if exp(t) < exp(gamma*exp(t)):
                    decided_gamma = gamma
                    break
            t_gamma.append(decided_gamma)
        return t_gamma

    def search_beta(self):
        tt = np.arange(10.5835, -0.001, -0.001)
        gg_pos = self.g(tt) # g(t) (t>0)
        gg_neg = self.g(-tt) # g(t) (t<0)
        
        peak_idx_pos = np.nanargmax(np.abs(gg_pos))
        peak_idx_neg = np.nanargmax(np.abs(gg_neg))

        peak_val_pos = gg_pos[peak_idx_pos]
        peak_val_neg = gg_neg[peak_idx_neg]

        print(f"peak_idx_pos = {peak_idx_pos}")
        print(f"peak_idx_neg = {peak_idx_neg}")
        
        t_thresh_pos = 0.0 # 仮の値
        idx_thresh_pos = 0
        for idx, t in enumerate(tt):
            if np.abs(gg_pos[idx] - (1/sqrt(2)) * peak_val_pos) < 0.01:
                t_thresh_pos = t
                idx_thresh_pos = idx
                break

        t_thresh_neg = 0.0
        idx_thresh_neg = 0
        for idx, t in enumerate(-tt):
            if np.abs(gg_neg[idx] - (1/sqrt(2)) * peak_val_neg) < 0.01:
                t_thresh_neg = t
                idx_thresh_neg = idx
                break
        
        print("#"*20)
        print(f"t_thresh_pos = {t_thresh_pos}")
        print(f"t_thresh_neg = {t_thresh_neg}")
        print("#"*20)
        
        if np.abs(t_thresh_pos) >= np.abs(t_thresh_neg):
            t_thresh = np.abs(t_thresh_pos)
            idx_thresh = idx_thresh_pos
        else:
            t_thresh = np.abs(t_thresh_neg)
            idx_thresh = idx_thresh_neg

       
        # if t_thresh == 0.0:
        #     raise Warning("Beta could not be found properly. Decide beta on your own.")

        t_alpha = self.calc_1st_approximation()
        t_gamma = self.calc_2nd_approximation()
        alpha_hat = t_alpha[idx_thresh]
        gamma_hat = t_gamma[idx_thresh]

        self.beta = pi / alpha_hat - gamma_hat
        print("beta = ", self.beta)

    def search_d(self):
        """ g(z) (z: 複素数)の特異点のうち、実軸に最も近いものを求め、
        実軸からの距離dを返す。

        [原理]
        変数変換φ(t)=tanh(pi/2sinh(t))により、g(z) = f(φ(z))φ'(z)の特異点について、
        φ'(z)の分母に着目することにより 0 <= d <= pi/2 の範囲にある。
        よって、f(φ(z)))の特異点を探せばよい。f(φ(z))の分母をgivenとすると、
        非線形方程式denominator_of_f(φ(z)) = 0 の解をニュートン法により求めれば、
        その解の虚部の絶対値とpi/2のうち小さい方がdである。

        Return
        ------
            d (float): 求めたパラメータd
        """
        # 初期値点生成
        img_part_list = np.linspace(-pi/2, pi/2, 100)
        z0_list = []
        for img_part in img_part_list:
            for real_part in np.linspace(-100, 100, 100):
                z0_list.append(real_part + img_part * 1.j)
        
        d = pi/2
        for z in z0_list:
            # 各初期値に関してニュートン法
            diff = 1e10
            z0 = z
            cnt = 0 #ループ100回を超えたら発散と判定
            flag = 0
            while diff > 1e-16 and cnt < 100:
                cnt += 1
                z_before = z
                if self.grad_f_phi_denom(z_before) == 0.:
                    flag = 1
                    break
                z = z_before - self.f_phi_denom(z_before) / self.grad_f_phi_denom(z_before)
                diff = np.abs(z - z_before)
            if flag:
                continue
            print(f"z0 = {z0} || z = {z}")
            if np.abs(z.imag) < d:
                d = np.abs(z.imag)

        self.d = d

        return d

    def m_to_h(self, m, d):
        """ 打ち切り点数mから離散化幅hを計算する。
        Arguments
        ---------
            m(int): 打ち切り点数
            d(float): search_d()で求めたパラメータd
        """
        return log((2*pi*d*m)/self.beta)/m

    def solve(self, m):
        """ 打ち切り点数mをもとに数値積分を実行する。

        Arguments
        --------
            m(int): 打ち切り点数
        
        Return
        ------
            value(float): 数値積分の結果
        """
        d = self.d
        print("################ DE Formula ##################")
        print("d =", d)
        print("m =", m)
        h = self.m_to_h(m, d)
        print("h =", h)
        value = 0.0
        print(f"STEP 000 || value = {value}")
        for j in range(-m, m):
            if value == np.inf:
                break
            value += 0.5 * (self.g(j*h) + self.g((j+1)*h)) * h
            print(f"STEP {str(j+m+1).zfill(3)} || value = {value}")
        return value
    
    def experiment(self, m_list, fig_name):
        self.search_d()
        self.search_beta()
        error_list = []
        for m in m_list:
            val = self.solve(m)
            error = np.abs(val - self.TRUTH)
            error_list.append(error)

        plt.figure()
        plt.plot(m_list, error_list)
        plt.scatter(m_list, error_list)
        plt.yscale("log")
        plt.xlabel("m点")
        plt.ylabel("誤差")
        plt.ylim(1e-17, 1e1)
        plt.title(f"{fig_name} | 打ち切り点数mによる誤差の変化")
        plt.savefig(f"{os.curdir}/images/{fig_name}.png")

        print("###########################")
        print(f"beta={self.beta} || d={self.d}")
        print("###########################")
    

def test(f, f_denom, f_phi, f_phi_denom, grad_f_phi_denom, truth, m_list, fig_name):  
    de = DEFormula(f, f_denom, f_phi, f_phi_denom, grad_f_phi_denom, truth)
    de.experiment(m_list=m_list, fig_name=fig_name)