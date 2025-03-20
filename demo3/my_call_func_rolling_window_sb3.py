import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.applications.stock_trading.stock_trading_rolling_window import (
    stock_trading_rolling_window,
)

import sys

# 添加项目根目录到路径
sys.path.append(".")

# 导入配置
from config import *
from config_tickers import DOW_30_TICKER

# 确保目录存在
RESULTS_DIR = os.path.join("demo3", RESULTS_DIR)
TRAINED_MODEL_DIR = os.path.join("demo3", TRAINED_MODEL_DIR)
for dir_path in [DATA_SAVE_DIR, RESULTS_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 设置滚动窗口参数
train_start_date = DEMO_TRAIN_START_DATE  # 使用配置文件中的起始日期
train_end_date = DEMO_TRAIN_END_DATE  # 训练截止到测试结束
trade_start_date = DEMO_TRADE_START_DATE  # 从测试开始进行交易
trade_end_date = DEMO_TRADE_END_DATE  # 使用配置文件中的交易结束日期

# 滚动窗口设置
rolling_window_length = 22  # 约一个月的交易日
if_store_actions = True  # 存储动作记录
if_store_result = True  # 存储结果

# 选择使用的算法
if_using_a2c = False
if_using_ppo = True
if_using_ddpg = True
if_using_td3 = False  # TD3训练较慢，可选择关闭
if_using_sac = False  # SAC训练较慢，可选择关闭

print(f"===== 滚动窗口交易策略 =====")
print(f"训练日期: {train_start_date} 到 {train_end_date}")
print(f"交易日期: {trade_start_date} 到 {trade_end_date}")
print(f"滚动窗口长度: {rolling_window_length}个交易日")
print(f"使用A2C: {if_using_a2c}, PPO: {if_using_ppo}, DDPG: {if_using_ddpg}")
print(f"使用TD3: {if_using_td3}, SAC: {if_using_sac}")
print(f"技术指标: {INDICATORS}")
print(f"股票: DOW_30_TICKER ({len(DOW_30_TICKER)}支股票)")

# 启动滚动窗口训练和交易
stock_trading_rolling_window(
    train_start_date,
    train_end_date,
    trade_start_date,
    trade_end_date,
    rolling_window_length,
    if_store_actions=if_store_actions,
    if_store_result=if_store_result,
    if_using_a2c=if_using_a2c,
    if_using_ddpg=if_using_ddpg,
    if_using_ppo=if_using_ppo,
    if_using_sac=if_using_sac,
    if_using_td3=if_using_td3,
)
