import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# samples = int(1e5)
# steps = int(1e3)
# # 敲入参与率 knocked-in participation rate k_p
# # 未敲入参与率 unknocked-in participation rate uk_p
# k_p = 1
# uk_p = 0.8
# tau = 1 # 1 year
# r = 0.03 # risk_free rate
# vol = 0.2
# s0 = 100
# K = 0.8

# BS model
# dS = rSdt + \sigma S dW

# np.random.seed(100)
# W = np.random.randn(samples, steps)
# Z = np.sqrt(tau/steps) * W
# S = np.zeros((samples, steps+1))
# S[:,0] = s0
# ko_indicator = np.zeros((samples,1))
#
# for i in range(1,steps+1):
#     S[:,i] = S[:,i-1] + r * S[:,i-1] * tau/steps + vol * S[:,i-1] * Z[:,i-1]
#
# ko_indicator[S.min(axis=1) < K*s0] = 1
#
# st = S[:,-1].reshape(-1,1)
# payoff = np.zeros((samples, 1))
# payoff[ko_indicator.astype(bool)] = k_p * (st[ko_indicator.astype(bool)]-s0)
# payoff[~ko_indicator.astype(bool)] = uk_p * (np.maximum(st[~ko_indicator.astype(bool)]-s0,0))
#
# price = np.mean(payoff) * np.exp(-r*tau)

class Airbag:
    def __init__(self,k_p=1, uk_p=0.8,
                 tau=1, r=0.03, vol=0.2, s0=100, K=0.8, samples=int(1e5), steps=int(1e3)):
        """
        :param k_p: 敲入参与率
        :param uk_p: 未敲入参与率
        :param tau: 到期时长 maturity
        :param r: 无风险利率
        :param vol: 波动率
        :param s0:
        :param K: [0,1] 敲入价格
        :param samples: 样本数量
        :param steps: 步数
        """
        self.k_p = k_p
        self.uk_p = uk_p
        self.tau = tau
        self.r = r
        self.vol = vol
        self.s0 = s0
        self.K = K
        self.reps = samples
        self.steps = steps

        self.simulation()

    def simulation(self):
        np.random.seed(100)
        W = np.random.randn(self.reps, self.steps)
        Z = np.sqrt(self.tau / self.steps) * W
        self.S = np.zeros((self.reps, self.steps + 1))
        self.S[:, 0] = self.s0
        self.ko_indicator = np.zeros((self.reps, 1))

        for i in range(1, self.steps + 1):
            self.S[:, i] = self.S[:, i - 1] + self.r * self.S[:, i - 1] * \
                           self.tau / self.steps + self.vol * self.S[:, i - 1] * Z[:, i - 1]

        self.ko_indicator[self.S.min(axis=1) < self.K * self.s0] = 1

        self.st = self.S[:, -1].reshape(-1, 1)
        self.payoff = np.zeros((self.reps, 1))
        self.payoff[self.ko_indicator.astype(bool)] = self.k_p * \
                                                      (self.st[self.ko_indicator.astype(bool)] - self.s0)
        self.payoff[~self.ko_indicator.astype(bool)] = self.uk_p * \
                                                       (np.maximum(self.st[~self.ko_indicator.astype(bool)] - self.s0, 0))

        self.price = np.mean(self.payoff) * np.exp(-self.r * self.tau)
        z = 1.96
        self.CI = z * np.std(self.payoff * np.exp(-self.r * self.tau),ddof=1) / np.sqrt(self.reps)
        print('MC价格: {:.3f}'.format(self.price))
        print("95% 置信区间为[{:.3f}, {:.3f}]".format(self.price-self.CI, self.price+self.CI))







