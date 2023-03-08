import numpy as np

S0 = 1
S_max = 4 * S0
s_in = 0.8 * S0
s_out = 1.03 * S0
R = 0.25
r = 0.03
n = 4

M = 360 * n
T = 360/365
vol = 0.2455
# ti = np.arange(3,13)/365
dt = T/M
delta_S = S_max/M


def UOP(M):
    f_mat = np.zeros((M+1,M+1))

    ## 纵轴s,从0--> M价格增大
    ## 横轴T,从0——>M时间增大


    f_mat[0,:] = np.array([S0 * np.exp(-r * (T - dt * i))
                           for i in range(f_mat.shape[1])])
    f_mat[:, -1] = np.array([max(S0 - delta_S * i, 0)
                             for i in range(f_mat.shape[0])])
    # 上边界
    f_mat[-1, :] = 0

    return f_mat

def DKOP(M):
    f_mat = np.zeros((M+1,M+1))

    ## 纵轴s,从0--> M价格增大
    ## 横轴T,从0——>M时间增大


    f_mat[0,:] = 0
    f_mat[:, -1] = np.array([max(S0 - delta_S * i, 0)
                             for i in range(f_mat.shape[0])])
    # 上边界
    f_mat[-1, :] = 0

    return f_mat

def gen_m1():

    M1 = np.zeros((M+1, M+1))
    M2 = np.zeros((M+1, M+1))
    # b =
    for j in range(1, M):
        aj = dt / 4 * ((vol * j) ** 2 - r * j)
        bj = -dt / 2 * ((vol * j) ** 2  + r)
        yj = dt / 4 * ((vol * j) ** 2  + r * j)

        M1[j,j] = 1 - bj
        M2[j,j] = 1 + bj
        if j != 1:
            M1[j,j-1] = -aj
            M2[j, j-1] = aj
        if j != M-1:
            M1[j, j+1] = -yj
            M2[j, j+1] = yj
    M1 = M1[1:-1, 1:-1]
    M2 = M2[1:-1, 1:-1]
    return M1,M2


def calculate_f_matrix(f_matrix, knock_in_flag=False):
    bar_out = int(np.ceil(s_out / delta_S))
    bar_in = int(np.floor(s_in / delta_S))
    m1, m2 = gen_m1()
    inverse_m1 = np.linalg.inv(m1)
    for i in range(M, 0, -1):
        # 迭代
        b = np.zeros(M-1)
        b[0] = dt/4 * ((vol * 1) ** 2 - r * 1) * (f_matrix[0,i-1] + f_matrix[0, i])
        b[-1] = dt/4 * ((vol * (M-1)) ** 2  + r * (M-1)) * (f_matrix[-1, i] + f_matrix[-1, i-1])
        Fi = f_matrix[1:M, i]
        Fi_1 = np.dot(inverse_m1 , (np.dot(m2 , Fi.reshape((-1, 1))) + b.reshape(-1,1)))

        if (i + 1) % (30 * n) == 0 and (i + 1) >= 30 * 3 * n:
            Fi_1[bar_out:] = 0
        if knock_in_flag:
            Fi_1[:bar_in] = 0

        Fi_1 = list(np.array(Fi_1.reshape(1, M - 1))[0])

        f_matrix[1:M, i-1]=Fi_1
    # 这一步取出S_t在网格中的位置，然后抽出结果，即为在该股价的期权价格。
    i = np.round(S0/delta_S, 0)
    return f_matrix[int(i), 0]


f_matrix_uop = UOP(M)
f_matrix_dkop = DKOP(M)
P_uop = calculate_f_matrix(f_matrix_uop)
P_dkop = calculate_f_matrix(f_matrix_dkop, True)
print("V3=%f" % (P_dkop - P_uop))