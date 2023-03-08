## Monte Carlo多种雪球产品定价(various_snowballs_pricing.py)
- 创建参数字典并导入、创建雪球类，然后运行类内函数simulation_main()即可定价。
- 不考虑对冲、交易成本、溢价等因素，仅从买方角度估值。
- 目前支持的雪球种类有 Stepdown 单降雪球， Vanilla 普通雪球， Airbag 安全气囊， Absolute_vanilla 绝对收益雪球
- e.g. params={'r':0.03, 'q':0.01, ...}; A = Vanilla(params); A.simulation_main()

## 给定价格估计公允票息(note_to_price.py)
- 根据卖方给出定价测算naive的票息。

## 雪球Delta加速
- 分别用Likelihood Ratio 和 Pathwise Estimator估计普通雪球的Delta, Likelihood Ratio相对收敛较快、速度较快
- 至少200万条路径才能给出较为稳定的delta，针对python运行速度较慢，用numba加速，每100万条path约用时8s。