import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
import datetime as dt
import seaborn as sb
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')


r = 0.04
q = 0.01
sigma = 0.15
tau = 1
S0 = 100
n_steps = 252
n_paths = 100000
knock_out = 1.12
knock_in = 0.85
note_return = 0.3
knock_out_interval = 21
step_t = tau / n_steps
return_list = []
S_list = []
price_list = []
knock_in_price = S0 * knock_in
knock_out_price = S0 * knock_out
knock_outs = 0
sqrt_t = np.sqrt(step_t)

def GBM_next(S, r, q, sigma, step_t):
    increment_S = np.exp((r - q - 0.5 * sigma ** 2) * step_t + sigma * sqrt_t * np.random.normal(0, 1))
    if increment_S > 1.1:
        return S * 1.1
    if increment_S < 0.9:
        return S * 0.9
    return S * increment_S

for j in range(n_paths):
    S = S0
    knock_in_flag = False
    knock_out_flag = False
    path_note_return = note_return
    for i in range(n_steps):
        S = GBM_next(S, r, q, sigma, step_t)
        if ((i + 1) % knock_out_interval == 0) and (S >= knock_out_price):
            knock_outs += 1
            path_return = path_note_return
            knock_out_flag = True
            break
        if S < knock_in_price:
            knock_in_flag = True
    if not knock_out_flag:
        if knock_in_flag:
            path_return = min(0, S / S0 - 1)
        else:
            path_return = path_note_return
    S_list.append(S)
    return_list.append(path_return)
    time_held = (i + 1) / n_steps
    price = return_list[-1] * S0 * time_held * np.exp(-r * time_held)
    price_list.append(price)
price_snowball = np.mean(price_list)
print("Simulated Price:%f" % price_snowball)


