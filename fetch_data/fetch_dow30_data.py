import os
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import sys
import datetime

# 导入配置
sys.path.append(".")
import config
import config_tickers


def create_directories():
    """创建必要的目录"""
    if not os.path.exists(config.DATA_SAVE_DIR):
        os.makedirs(config.DATA_SAVE_DIR)
    if not os.path.exists(config.TRAINED_MODEL_DIR):
        os.makedirs(config.TRAINED_MODEL_DIR)
    if not os.path.exists(config.TENSORBOARD_LOG_DIR):
        os.makedirs(config.TENSORBOARD_LOG_DIR)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)


def fetch_data():
    """获取股票数据"""
    print(f"开始从Yahoo Finance下载数据...")
    # 这里使用DOW_30_TICKER股票列表
    df = YahooDownloader(
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRADE_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()

    # 确保数据目录存在
    if not os.path.exists(config.DATA_SAVE_DIR):
        os.makedirs(config.DATA_SAVE_DIR)

    # 保存原始数据
    df.to_csv(f"{config.DATA_SAVE_DIR}/dow_30_raw_data.csv", index=False)
    print(f"原始数据已保存到 {config.DATA_SAVE_DIR}/dow_30_raw_data.csv")

    return df


if __name__ == "__main__":
    # 创建必要目录
    create_directories()

    # 获取数据
    df = fetch_data()

    # 显示基本统计信息
    print("\n数据基本统计信息:")
    print(f"数据时间范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"股票数量: {df['tic'].nunique()}")
    print(f"总记录数: {len(df)}")

    # 显示前几行数据
    print("\n数据样例:")
    print(df.head())
