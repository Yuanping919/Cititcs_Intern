import pandas as pd
import time
import datetime as dt
import numpy as np
import chinese_calendar as cc
import matplotlib.pyplot as plt
from numba import jit
plt.style.use('ggplot')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class Snowballs(object):
    def __init__(self, params):
        self.r = params['r']
        self.q = params['q']
        self.sigma = params['sigma']
        self.tau = params['tau']
        self.S0 = params['S0']
        self.moved_S = params['moved_S']
        self.n_paths = params['n_paths']
        self.option_duration_months = params['option_duration_months']
        self.start_date = params['start_date']
        self.delta_S = params['delta_percent'] * self.S0
        # 如果合约按自然月观察则用get_knockout_dates获取n_steps和观察日
        # self.n_steps = params['n_steps']
        self.random_normals = None
        self.S_list = []
        self.return_list = []
        self.duration_list = []
        self.price_list = []


    def get_knockout_dates(self):
        knock_out_observe_day = self.start_date.day
        month_pointer = self.start_date.month
        year_pointer = self.start_date.year
        date_pointer = self.start_date
        self.knock_out_index_list = [0]
        self.knock_out_date_list = [date_pointer]
        self.printed_knock_out_dates = [date_pointer.strftime("%Y-%m-%d")]
        for i in range(self.option_duration_months):
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
            self.knock_out_index_list.append(self.knock_out_index_list[-1] + index_increment)
            self.knock_out_date_list.append(date_pointer)
            self.printed_knock_out_dates.append(date_pointer.strftime("%Y-%m-%d"))
        self.n_steps = self.knock_out_index_list[-1]
        self.step_t = self.tau / self.n_steps
        self.sqrt_t = np.sqrt(self.step_t)


    def if_knock_out_date(self, step_i):
        judge_knockout = self.knock_out_index_list[self.knock_out_pointer] == (step_i + 1)
        if judge_knockout:
            self.knock_out_pointer += 1
        return judge_knockout


    def GBM_generator(self):
        # Antithetic Variates Method
        self.random_normals = np.random.normal(0, 1, size=(self.n_paths // 2, self.n_steps))
        self.random_normals = np.concatenate([self.random_normals, -self.random_normals])


    def GBM_next(self, S, path_i, step_i):
        if self.random_normals is None:
            increment_S = np.exp((self.r - self.q - 0.5 * self.sigma ** 2) * \
                                 self.step_t + self.sigma * self.sqrt_t * np.random.normal(0, 1))
        else:
            increment_S = np.exp((self.r - self.q - 0.5 * self.sigma ** 2) * \
                                 self.step_t + self.sigma * self.sqrt_t * self.random_normals[path_i][step_i])
        if increment_S > 1.1:
            return S * 1.1
        if increment_S < 0.9:
            return S * 0.9
        return S * increment_S

    def single_path(self, path_i):
        raise Exception("Simulation not implemented!")

    def init_path(self, S):
        self.S = S
        self.knock_out_pointer = 1


    def simulation_main(self):
        self.get_knockout_dates()
        self.GBM_generator()
        for path_i in range(self.n_paths):
            if path_i % 100000 == 0:
                print("Simulating path %d" % path_i)
            self.init_path(self.S0)
            path_return, duration, price = self.single_path(path_i)
            self.S_list.append(self.S)
            self.return_list.append(path_return)
            self.duration_list.append(duration)
            self.price_list.append(price)
        self.simulated_results = np.mean(self.price_list)
        print("Simulated target is %f" % self.simulated_results)

    def calc_delta_pathwise(self):
        self.get_knockout_dates()
        self.GBM_generator()
        self.delta_list = []
        for path_i in range(self.n_paths):
            if path_i % 100000 == 0:
                print("Simulating path %d" % path_i)
            S_minus = self.moved_S - self.delta_S
            S_plus = self.moved_S + self.delta_S
            self.init_path(S_minus)
            _, _, snowball_price_minus = self.single_path(path_i)
            self.init_path(S_plus)
            _, _, snowball_price_plus = self.single_path(path_i)
            cur_delta = (snowball_price_plus - snowball_price_minus) / (2 * self.delta_S)
            self.delta_list.append(cur_delta)
        self.delta_results = np.mean(self.delta_list)
        print("Simulated delta is %f" % self.delta_results)

    def calc_delta_likelihood(self):
        self.get_knockout_dates()
        self.GBM_generator()
        self.delta_list = []
        for path_i in range(self.n_paths):
            if path_i % 100000 == 0:
                print("Simulating path %d" % path_i)
            self.init_path(self.moved_S)
            likelihood_ratio = self.random_normals[path_i][0] \
                                   / (self.S * self.sigma * self.sqrt_t)
            _, _, payoff = self.single_path(path_i)
            cur_delta = payoff * likelihood_ratio
            self.delta_list.append(cur_delta)
        self.delta_results = np.mean(self.delta_list)
        print("Simulated delta is %f" % self.delta_results)

class Vanilla(Snowballs):
    def __init__(self, params):
        super().__init__(params)
        self.knock_in = params['knock_in']
        self.knock_out = params['knock_out']
        self.note_return = params['note_return']
        # 使用自然月日期则不用interval
        self.knock_out_interval = params['knock_out_interval']
        self.knock_out_price = self.S0 * self.knock_out
        self.knock_in_price = self.knock_in * self.S0
        self.knockout_numbers = 0

    def single_path(self, path_i):
        knock_in_flag = False
        knock_out_flag = False
        path_note_return = self.note_return
        for step_i in range(self.n_steps):
            self.S = self.GBM_next(self.S, path_i, step_i)
            if self.if_knock_out_date(step_i) and \
                    (self.S >= self.knock_out_price):
                path_return = path_note_return
                knock_out_flag = True
                break
            if self.S < self.knock_in_price:
                knock_in_flag = True
        if not knock_out_flag:
            if knock_in_flag:
                path_return = min(0, self.S / self.S0 - 1)
            else:
                path_return = path_note_return
        duration = (step_i + 1) / self.n_steps
        price = path_return * self.S0 * duration * np.exp(-self.r * duration)
        return path_return, duration, price



class Airbag(Snowballs):
    def __init__(self, params):
        super().__init__(params)
        self.up_barrier = params['up_barrier']
        self.knock_in = params['knock_in']
        self.knock_in_price = self.knock_in * self.S0
        self.up_barrier_price = self.up_barrier * self.S0

    def single_path(self, path_i):
        knock_in_flag = False
        for step_i in range(self.n_steps):
            self.S = self.GBM_next(self.S, path_i, step_i)
            if self.S < self.knock_in_price:
                knock_in_flag = True
        if knock_in_flag:
            path_return = min(0, self.S / self.S0 - 1)
        else:
            path_return = min(self.up_barrier_price, max(0, self.S / self.S0 - 1))
        price = path_return * self.S0 * np.exp(-self.r)
        duration = (step_i + 1) / self.n_steps
        return path_return, duration, price


class Stepdown(Snowballs):
    def __init__(self, params):
        super().__init__(params)
        self.knock_in = params['knock_in']
        self.knock_out = params['knock_out']
        self.note_return = params['note_return']
        self.knock_out_interval = params['knock_out_interval']
        self.return_stepdown_rate = params['return_stepdown_rate']
        self.knock_out_price = self.S0 * self.knock_out
        self.knock_in_price = self.knock_in * self.S0

    def single_path(self, path_i):
        knock_in_flag = False
        knock_out_flag = False
        path_note_return = self.note_return
        for step_i in range(self.n_steps):
            self.S = self.GBM_next(self.S, path_i, step_i)
            if self.if_knock_out_date(step_i):
                path_note_return -= self.return_stepdown_rate
                if self.S >= self.knock_out_price:
                    path_return = path_note_return
                    knock_out_flag = True
                    break
            if self.S < self.knock_in_price:
                knock_in_flag = True
        if not knock_out_flag:
            if knock_in_flag:
                path_return = min(0, self.S / self.S0 - 1)
            else:
                path_return = path_note_return
        duration = (step_i + 1) / self.n_steps
        price = path_return * self.S0 * duration * np.exp(-self.r * duration)
        return path_return, duration, price

class Absolute_vanilla(Vanilla):
    def __init__(self, params):
        super().__init__(params)

    def single_path(self, path_i):
        knock_in_flag = False
        knock_out_flag = False
        path_note_return = self.note_return
        for step_i in range(self.n_steps):
            self.S = self.GBM_next(self.S, path_i, step_i)
            if self.if_knock_out_date(step_i) and \
                    (self.S >= self.knock_out_price):
                path_return = path_note_return
                knock_out_flag = True
                break
            if self.S < self.knock_in_price:
                knock_in_flag = True
        if not knock_out_flag:
            if knock_in_flag:
                path_return = min(0, self.S / self.S0 - 1)
            else:
                path_return = path_note_return
        duration = (step_i + 1) / self.n_steps
        price = path_return * self.S0 * np.exp(-self.r * duration)
        return path_return, duration, price


class European_put(Snowballs):
    def __init__(self, params):
        super().__init__(params)
        self.K = params['K'] * self.S0

    def single_path(self, path_i):
        self.S = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma ** 2) * \
                                 self.tau + self.sigma * np.sqrt(self.tau) * np.random.normal(0, 1))
        price = max(self.K - self.S, 0)
        path_return = self.S / self.S0 - 1
        return path_return, None, price

    def short_put(self):
        short_put_list = []
        for i in range(1000):
            final_loss = max(0, self.K - self.S_list[i])
            short_put_return = 5.062737 - final_loss
            short_put_list.append(short_put_return)
        return short_put_list

params = {'r': 0.03,
          'q': 0.0,
          'sigma': 0.2455,
          'tau': 1,
          'S0': 1,
          'moved_S': 1,
          'n_paths': 100000,
          'delta_percent': 0.0001,
          'n_steps': 252,
          'start_date': dt.datetime(2022, 1, 4),
          'option_duration_months': 12,
          'knock_in': 0.8,
          'K': 1,
          'knock_out': 1.03,
          'note_return': 0.25,
          'up_barrier': 1.45, # for airbags
          'return_stepdown_rate': 0.02,
          'knock_out_interval': 21}


t1 = time.time()
# A = Stepdown(params)
# A.simulation_main()
#
A = Vanilla(params)
A.simulation_main()
# A.calc_delta_pathwise()
# A.calc_delta_likelihood()
#
# A = Airbag(params)
# A.simulation_main()

# A = Absolute_vanilla(params)
# A.simulation_main()

# A = Snowballs(params)
# A.get_knockout_dates()
# print(A.knock_out_index_list)
# print(','.join(A.printed_knock_out_dates))

# A = European_put(params)
# A.simulation_main()
# short_returns = A.short_put()

# import seaborn as sb
# plt.title("S=100,票息0.2,敲入0.85,敲出1.03的雪球模拟收益")
# sb.distplot(A.price_list[:10000])
# plt.title('卖出一年期100% Put收益分布')
# sb.distplot(short_returns)


# new_arr = {}
# for i in range(15):
#     S_arr = [A.S0]
#     for z in range(A.n_steps):
#         S_arr.append(A.GBM_next(S_arr[-1], i, z))
#     path_str = 'path %s' % str(i + 1)
#     new_arr[path_str] = S_arr
# paths_df = pd.DataFrame(new_arr)
# plt.title('Monte Carlo模拟的一年期路径')
# plt.plot(paths_df)

t2 = time.time()
print(t2 - t1)