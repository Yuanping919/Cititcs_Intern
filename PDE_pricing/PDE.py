import numpy as np

class Callable:
    def __init__(self, s0=1, R=0.25, r=0.03, T=360/365, vol=0.2455, n=4):
        self.s0 = s0
        self.s_max = 4 * self.s0
        self.s_in = 0.8 * s0
        self.s_out = 1.03 * s0
        self.R = R
        self.r = r
        self.T = T
        self.vol = vol
        self.n = n
        self.M = 360 * self.n
        # ti = np.arange(3,13)/365
        self.dt = self.T / self.M
        self.delta_S = self.s_max / self.M
        # self.count = 0

        self.otu_bound()
        self.dot_bound()
        self.uop_bound()
        self.dkop_bound()

    def gen_m(self):
        # generate M1,M2
        M1 = np.zeros((self.M + 1,self.M + 1))
        M2 = np.zeros((self.M + 1, self.M + 1))
        # b =
        for j in range(1, self.M):
            aj = self.dt / 4 * ((self.vol * j) ** 2 - self.r * j)
            bj = -self.dt / 2 * ((self.vol * j) ** 2 + self.r)
            yj = self.dt / 4 * ((self.vol * j) ** 2 + self.r * j)

            M1[j, j] = 1 - bj
            M2[j, j] = 1 + bj
            if j != 1:
                M1[j, j - 1] = -aj
                M2[j, j - 1] = aj
            if j != self.M - 1:
                M1[j, j + 1] = -yj
                M2[j, j + 1] = yj
        self.M1 = M1[1:-1, 1:-1]
        self.M2 = M2[1:-1, 1:-1]

    def otu_bound(self):
        """"""
        """上涨生效触碰期权 One-Touch Up"""

        self.otu_mat = np.zeros((self.M + 1, self.M + 1))

        ## 纵轴s,从0--> M价格增大
        ## 横轴T,从0——>M时间增大

        self.otu_mat[0, :] = 0
        self.otu_mat[:, -1] = 0
        # 上边界
        self.otu_mat[-1, self.n * 30 * np.arange(3, 13)] = self.R * 30 * np.arange(3, 13) / 365 * self.s0

    def dot_bound(self):
        """"""
        """双边失效触碰期权 Double No-Touch"""
        """双边触碰生效期权 Double One-Touch"""
        self.dnt_mat = np.zeros((self.M + 1, self.M + 1))
        bar_out = int(np.ceil((self.s_out) / self.delta_S))
        ## 纵轴s,从0--> M价格增大
        ## 横轴T,从0——>M时间增大
        # f(T,S) = 0
        self.dnt_mat[:, -1] = 0
        # f_mat[bar_out:,-1] = R * T * s0
        "# f(t, S_min) = R * T * exp{-r(T-t)}"
        self.dnt_mat[0, :] = self.R * self.T * \
                             np.exp(-self.r * (self.T - np.arange(self.M + 1) / self.M * self.T)) * self.s0
        # f(t_i, S_max) = R * T * exp{-r(T-t)}
        self.dnt_mat[-1, self.n * 30 * np.arange(3, 13)] = self.R * self.T * \
                                                           np.exp(-self.r * (self.T - np.arange(3, 13) * 30 / 365)) * self.s0

    def uop_bound(self):
        """"""
        """上涨失效看跌障碍期权 Up and Out Put"""
        self.uop_mat = np.zeros((self.M + 1, self.M + 1))
        bar = int(np.ceil(self.s_out / self.delta_S))
        ## 纵轴s,从0--> M价格增大
        ## 横轴T,从0——>M时间增大

        self.uop_mat[0, :] = 0

        self.uop_mat[:, -1] = np.maximum(self.s0 - np.arange(self.M + 1) * self.delta_S, 0)
        # f_mat[bar,-1] = 0
        # 上边界
        self.uop_mat[-1, :] = self.s0 * np.exp(-self.r * (1 - np.arange(self.M + 1) / self.M) * self.T)

    def dkop_bound(self):
        """"""
        """双边失效看跌障碍期权 Double Knock-Out Put，DKOP"""
        self.dkop_mat = np.zeros((self.M + 1, self.M + 1))
        bar = int(np.ceil((self.s_out) / self.delta_S))
        ## 纵轴s,从0--> M价格增大
        ## 横轴T,从0——>M时间增大

        self.dkop_mat[0, :] = 0

        self.dkop_mat[:, -1] = np.maximum(self.s0 - np.arange(self.M + 1) * self.delta_S, 0)
        # f_mat[bar,-1] = 0
        # 上边界
        self.dkop_mat[-1, :] = 0

    def iter_otu(self, i):
        b = np.zeros(self.M - 1)
        b[0] = self.dt / 4 * ((self.vol * 1) ** 2 - self.r * 1) * \
                   (self.otu_mat[0, i - 1] + self.otu_mat[0, i])
        b[-1] = self.dt / 4 * ((self.vol * (self.M - 1)) ** 2 + self.r * (self.M - 1)) * \
                (self.otu_mat[-1, i] + self.otu_mat[-1, i - 1])
        Fi = self.otu_mat[1:self.M, i]
        Fi_1 = np.dot(self.inverse_m1, (np.dot(self.M2, Fi.reshape((-1, 1))) + b.reshape(-1, 1)))

        if self.otu_mat[-1, i - 1] != 0:
            # self.count += 1
            Fi_1[self.bar_out:] = self.R * i * self.dt * self.s0

        Fi_1 = list(np.array(Fi_1.reshape(1, self.M - 1))[0])

        self.otu_mat[1:self.M, i - 1] = Fi_1

    def iter_dnt(self, i):
        b = np.zeros(self.M - 1)
        b[0] = self.dt / 4 * ((self.vol * 1) ** 2 - self.r * 1) * \
               (self.dnt_mat[0, i - 1] + self.dnt_mat[0, i])
        b[-1] = self.dt / 4 * ((self.vol * (self.M - 1)) ** 2 + self.r * (self.M - 1)) * \
                (self.dnt_mat[-1, i] + self.dnt_mat[-1, i - 1])
        Fi = self.dnt_mat[1:self.M, i]
        Fi_1 = np.dot(self.inverse_m1, (np.dot(self.M2, Fi.reshape((-1, 1))) + b.reshape(-1, 1)))

        if (i-1)%(30*self.n)==0 and i-1 >= 30 * 3 * self.n:
            Fi_1[self.bar_out:] = self.R * self.T * np.exp(-self.r * (1-(i-1)/self.M) * self.T) * self.s0
        if i-1 >= 30 * 3 * self.n:
            Fi_1[:self.bar_in] = self.R * self.T * np.exp(-self.r * (1-(i-1)/self.M) * self.T) * self.s0
        Fi_1 = list(np.array(Fi_1.reshape(1, self.M - 1))[0])
        self.dnt_mat[1:self.M, i-1]=Fi_1

    def iter_uop(self,i):
        b = np.zeros(self.M - 1)
        b[0] = self.dt / 4 * ((self.vol * 1) ** 2 - self.r * 1) * \
               (self.uop_mat[0, i - 1] + self.uop_mat[0, i])
        b[-1] = self.dt / 4 * ((self.vol * (self.M - 1)) ** 2 + self.r * (self.M - 1)) * \
                (self.uop_mat[-1, i] + self.uop_mat[-1, i - 1])
        Fi = self.uop_mat[1:self.M, i]
        Fi_1 = np.dot(self.inverse_m1, (np.dot(self.M2, Fi.reshape((-1, 1))) + b.reshape(-1, 1)))

        if (i-1)%(30 * self.n)==0 and i-1 >= 30 * 3 * self.n:
            Fi_1[self.bar_out:] = 0

        Fi_1 = list(np.array(Fi_1.reshape(1, self.M - 1))[0])
        self.uop_mat[1:self.M, i - 1] = Fi_1

    def iter_dkop(self, i):
        b = np.zeros(self.M - 1)
        b[0] = self.dt / 4 * ((self.vol * 1) ** 2 - self.r * 1) * \
               (self.dkop_mat[0, i - 1] + self.dkop_mat[0, i])
        b[-1] = self.dt / 4 * ((self.vol * (self.M - 1)) ** 2 + self.r * (self.M - 1)) * \
                (self.dkop_mat[-1, i] + self.dkop_mat[-1, i - 1])
        Fi = self.dkop_mat[1:self.M, i]
        Fi_1 = np.dot(self.inverse_m1, (np.dot(self.M2, Fi.reshape((-1, 1))) + b.reshape(-1, 1)))

        if (i-1)%(30 * self.n)==0 and i-1 >= 30 * 3 * self.n:
            Fi_1[self.bar_out:] = 0
        if i - 1 >= 30 * 3 * self.n:
            Fi_1[:self.bar_in] = 0
        Fi_1 = list(np.array(Fi_1.reshape(1, self.M - 1))[0])

        self.dkop_mat[1:self.M, i-1]=Fi_1


    def calculate(self):
        self.bar_out = int(np.ceil(self.s_out / self.delta_S))
        self.bar_in = int(np.floor(self.s_in/self.delta_S))

        self.gen_m()
        self.inverse_m1 = np.linalg.inv(self.M1)
        for i in range(self.M, 0, -1):
            # 迭代
            self.iter_dnt(i)
            self.iter_otu(i)
            self.iter_uop(i)
            self.iter_dkop(i)
        # 这一步取出S_t在网格中的位置，然后抽出结果，即为在该股价的期权价格。
        pos = np.round(self.s0 / self.delta_S, 0)
        self.otu_p = self.otu_mat[int(pos), 0]
        self.dot_p = self.dnt_mat[int(pos), 0]
        self.dnt_p = self.R * self.T * np.exp(-self.r * self.T) - self.dot_p
        self.uop_p = self.uop_mat[int(pos), 0]
        self.dkop_p = self.dkop_mat[int(pos), 0]

        self.price = self.otu_p + self.dnt_p + self.dkop_p - self.uop_p

    def summary(self):
        print("上涨生效触碰期权的价格为{:.5f}".format(self.otu_p))

        print("双边触碰失效期权的价格为{:.5f}".format(self.dnt_p))
        print("双边触碰生效期权的价格为{:.5f}".format(self.dot_p))

        print("双边失效看跌障碍期权的价格为{:.5f}".format(self.dkop_p))
        print("上涨失效看跌障碍期权的价格为{:.5f}".format(self.uop_p))
        print("======================================")
        print("雪球结构产品的价格为{:.5f}".format(self.price))

option = Callable()
option.calculate()
option.summary()