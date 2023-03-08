import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
import datetime as dt
import seaborn as sb
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from numba import jit
import time
import chinese_calendar as cc
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

r = 0.03
q = 0.00
sigma = 0.2455
tau = 1
S0 = 1
n_paths = 1000000
knock_out = 1.03
knock_in = 0.80
note_return = 0.25
start_date = dt.datetime(2022, 1, 4)
option_duration_months = 12
delta_S = 0.01
knock_in_price = S0 * knock_in
knock_out_price = S0 * knock_out


def GBM_next(S, r, q, sigma, step_t):
    increment_S = np.exp((r - q - 0.5 * sigma ** 2) * step_t + sigma * sqrt_t * np.random.normal(0, 1))
    if increment_S > 1.1:
        return S * 1.1
    if increment_S < 0.9:
        return S * 0.9
    return S * increment_S
def GBM_calc(S, r, q, sigma, step_t, z):
    increment_S = np.exp((r - q - 0.5 * sigma ** 2) * step_t + sigma * sqrt_t * z)
    if increment_S > 1.1:
        return S * 1.1
    if increment_S < 0.9:
        return S * 0.9
    return S * increment_S


def get_knockout_dates(start_date, option_duration_months):
    knock_out_observe_day = start_date.day
    month_pointer = start_date.month
    year_pointer = start_date.year
    date_pointer = start_date
    knock_out_index_list = [0]
    knock_out_date_list = [date_pointer]
    printed_knock_out_dates = [date_pointer.strftime("%Y-%m-%d")]
    for i in range(option_duration_months):
        last_date = date_pointer
        if month_pointer == 12:
            month_pointer = 1
            year_pointer += 1
        else:
            month_pointer += 1
        date_pointer = dt.datetime(year_pointer, month_pointer, knock_out_observe_day)
        while cc.is_holiday(date_pointer):
            date_pointer += dt.timedelta(days=1)
        index_increment = len(cc.get_workdays(last_date, date_pointer)) - 1
        knock_out_index_list.append(knock_out_index_list[-1] + index_increment)
        knock_out_date_list.append(date_pointer)
        printed_knock_out_dates.append(date_pointer.strftime("%Y-%m-%d"))
    n_steps = knock_out_index_list[-1]
    step_t = tau / n_steps
    sqrt_t = np.sqrt(step_t)
    return  n_steps, step_t, sqrt_t, knock_out_index_list

n_steps, step_t, sqrt_t, knock_out_index_list = get_knockout_dates(start_date, option_duration_months)
knock_out_index_list = [knock_out_index_list[0]] + knock_out_index_list[3:]
knock_out_index_arr = np.array(knock_out_index_list)


def single_path(S, S0, r, q, sigma, step_t, note_return, knock_out_index_arr, z):
    knock_outs = 0
    knock_in_flag = False
    knock_out_flag = False
    knock_out_ptr = 1
    for i in range(n_steps):
        S = GBM_calc(S, r, q, sigma, step_t, z[i])
        if ((i + 1) == knock_out_index_arr[knock_out_ptr]):
            knock_outs += 1
            if (S >= knock_out_price):
                R = note_return
                knock_out_flag = True
                break
        if S < knock_in_price:
            knock_in_flag = True
    if not knock_out_flag:
        if knock_in_flag:
            R = min(0, S / S0 - 1)
        else:
            R = note_return
    time_held = (i + 1) / n_steps
    snowball_price = R * S0 * time_held * np.exp(-r * time_held)
    return snowball_price

@jit
def calc_delta_likelihood(S, S0, r, q, sigma, step_t, note_return, knock_out_index_arr, n_paths=n_paths, n_steps=n_steps):
    delta_sum = 0
    S_org = S
    random_normals = np.random.normal(0, 1, size=(n_paths, n_steps))
    for path_i in range(n_paths):
        likelihood_ratio = random_normals[path_i][0] \
                               / (S * sigma * sqrt_t)

        S = S_org
        knock_in_flag = False
        knock_out_flag = False
        knock_out_ptr = 1

        for i in range(n_steps):
            increment_S = np.exp((r - q - 0.5 * sigma ** 2) * step_t + sigma * sqrt_t * random_normals[path_i][i])
            if increment_S > 1.1:
                S = S * 1.1
            elif increment_S < 0.9:
                S = S * 0.9
            else:
                S = S * increment_S
            if ((i + 1) == knock_out_index_arr[knock_out_ptr]):
                knock_out_ptr += 1
                if (S >= knock_out_price):
                    R = note_return
                    knock_out_flag = True
                    break
            if S < knock_in_price:
                knock_in_flag = True
        if not knock_out_flag:
            if knock_in_flag:
                R = min(0, S / S0 - 1)
            else:
                R = note_return
        time_held = (i + 1) / n_steps
        payoff = R * S0 * time_held * np.exp(-r * time_held)


        cur_delta = payoff * likelihood_ratio
        delta_sum += cur_delta
    delta_results = delta_sum / n_paths
    # print("Simulated delta is %f" % delta_results)
    return delta_results

@jit
def calc_price(S, S0, r, q, sigma, step_t, note_return, knock_out_index_arr, n_paths=n_paths, n_steps=n_steps):
    delta_sum = 0
    S_org = S
    random_normals = np.random.normal(0, 1, size=(n_paths, n_steps))
    for path_i in range(n_paths):
        S = S_org
        knock_in_flag = False
        knock_out_flag = False
        knock_out_ptr = 1

        for i in range(n_steps):
            increment_S = np.exp((r - q - 0.5 * sigma ** 2) * step_t + sigma * sqrt_t * random_normals[path_i][i])
            S = S * increment_S
            if ((i + 1) == knock_out_index_arr[knock_out_ptr]):
                knock_out_ptr += 1
                if (S >= knock_out_price):
                    R = note_return
                    knock_out_flag = True
                    break

        if not knock_out_flag:
            R = 0
        time_held = (i + 1) / n_steps
        payoff = R * S0 * time_held * np.exp(-r * time_held)

        likelihood_ratio = 1
        cur_delta = payoff * likelihood_ratio
        delta_sum += cur_delta
    delta_results = delta_sum / n_paths
    # print("Simulated delta is %f" % delta_results)
    return delta_results

print("Simulated P:%f" % (calc_price(S0, S0, r, q, sigma, step_t, note_return, knock_out_index_arr)))

# plot_interval = (knock_out_price - knock_in_price) / 50
# plot_scale = np.arange(knock_in_price - plot_interval * 10,
#                        knock_out_price + plot_interval * 10, plot_interval)
# delta_list = []
# t1 = time.time()
# for i in range(len(plot_scale)):
#     delta_simulated = calc_delta_likelihood(plot_scale[i], S0, r, q, sigma, step_t, note_return, knock_out_index_arr)
#     delta_list.append(delta_simulated)
#     print("Simulated Delta:%f" % (delta_simulated))
# t2 = time.time()
# print("Time elapsed=%f" % (t2 - t1))
# delta_series = pd.Series(dict(zip(plot_scale, delta_list)))
# plt.title("雪球的Delta相对当前价格变化")
# delta_series.plot(figsize=(16, 8))

delta_list = \
    [1.086430,1.061535,1.094032,1.069718,1.126739,1.133195,1.132883,1.227338
,1.312651,1.381589,1.546256,1.576032,1.602418,1.603092,1.589231,1.601582
,1.556516,1.543122,1.529394,1.508220,1.467077,1.444266,1.371193,1.370038
,1.307384,1.279775,1.242878,1.206851,1.151691,1.121950,1.062452,1.035455
,0.978381,0.928356,0.881536,0.834841,0.767375,0.752810,0.669855,0.627395
,0.579560,0.530721,0.465649,0.434350,0.385571,0.336541,0.272242,0.240851
,0.200833,0.140943,0.099394,0.067154,0.016123,-0.007161,-0.042013,-0.069861
,-0.084255,-0.119317,-0.128062,-0.145802,-0.155086,-0.174479,-0.175735
,-0.171011,-0.174422,-0.175568,-0.186828,-0.173612,-0.166327,-0.158180]

# calculate Delta with common random number method (DEPRECATED)
# def calc_delta(S, S0, r, q, sigma, step_t, note_return, delta_S=delta_S):
#     S_minus_list, return_minus = [], []
#     S_plus_list, return_plus = [], []
#     delta_list = []
#     for j in range(n_paths):
#         S_minus = S - delta_S
#         S_plus = S + delta_S
#         z = np.random.normal(0, 1, size=n_steps)
#         snowball_price_minus = single_path(S_minus, S0, r, q, sigma, step_t, note_return, z, S_minus_list, return_minus)
#         snowball_price_plus = single_path(S_plus, S0, r, q, sigma, step_t, note_return, z, S_plus_list, return_plus)
#         pos_delta = (snowball_price_plus - snowball_price_minus) / (2 * delta_S)
#         snowball_price_minus = single_path(S_minus, S0, r, q, sigma, step_t, note_return, -z, S_minus_list, return_minus)
#         snowball_price_plus = single_path(S_plus, S0, r, q, sigma, step_t, note_return, -z, S_plus_list, return_plus)
#         neg_delta = (snowball_price_plus - snowball_price_minus) / (2 * delta_S)
#         avg_delta = (pos_delta + neg_delta) / 2
#         delta_list.append(avg_delta)
#     return delta_list