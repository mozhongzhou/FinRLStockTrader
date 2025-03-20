import os
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import itertools
import sys
import datetime
import argparse

# 导入配置
sys.path.append(".")
import config
import config_tickers


def load_data(ticker_type="dow30"):
    """
    加载原始数据

    Parameters:
    ticker_type (str): 指数类型，可选 "dow30", "sp500", "nasdaq100" 或 "all"

    Returns:
    df (DataFrame): 包含股票数据的DataFrame
    """
    print(f"加载{ticker_type}数据...")

    # 根据指数类型选择文件路径
    if ticker_type == "dow30":
        data_path = f"{config.DATA_SAVE_DIR}/dow_30_raw_data.csv"
    elif ticker_type == "sp500":
        data_path = f"{config.DATA_SAVE_DIR}/sp_500_raw_data.csv"
    elif ticker_type == "nasdaq100":
        data_path = f"{config.DATA_SAVE_DIR}/nasdq_100_raw_data.csv"
    elif ticker_type == "all":
        # 如果选择"all"，则合并所有指数数据
        dow30_path = f"{config.DATA_SAVE_DIR}/dow_30_raw_data.csv"
        sp500_path = f"{config.DATA_SAVE_DIR}/sp_500_raw_data.csv"
        nasdaq100_path = f"{config.DATA_SAVE_DIR}/nasdq_100_raw_data.csv"

        dfs = []
        if os.path.exists(dow30_path):
            dfs.append(pd.read_csv(dow30_path))
        if os.path.exists(sp500_path):
            dfs.append(pd.read_csv(sp500_path))
        if os.path.exists(nasdaq100_path):
            dfs.append(pd.read_csv(nasdaq100_path))

        if not dfs:
            raise FileNotFoundError(f"未找到任何指数数据文件")

        df = pd.concat(dfs)
        # 去除重复的股票数据
        df = df.drop_duplicates(subset=["date", "tic"])
        print(f"合并后共有{len(df)}行数据，{df['tic'].nunique()}支股票")
        return df
    else:
        raise ValueError(f"不支持的指数类型: {ticker_type}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}，请先运行fetch_data脚本")

    df = pd.read_csv(data_path)
    print(f"加载了{len(df)}行数据，{df['tic'].nunique()}支股票")
    return df


def ensure_data_continuity(df):
    """
    确保数据连续性，处理缺失值和生成完整的日期-股票组合

    Parameters:
    df (DataFrame): 原始股票数据

    Returns:
    df_full (DataFrame): 处理后的完整数据
    """
    print("确保数据连续性...")

    # 获取唯一的股票代码和日期
    list_ticker = df["tic"].unique().tolist()
    list_date = list(pd.date_range(df["date"].min(), df["date"].max()).astype(str))

    # 生成所有日期和股票的组合
    combination = list(itertools.product(list_date, list_ticker))

    # 创建一个包含所有可能组合的DataFrame
    df_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        df, on=["date", "tic"], how="left"
    )

    # 只保留原始数据中存在的日期
    df_full = df_full[df_full["date"].isin(df["date"])]

    # 排序
    df_full = df_full.sort_values(["date", "tic"])

    # 填充缺失值
    df_full = df_full.fillna(0)

    print(f"处理后共有{len(df_full)}行数据")
    return df_full


def add_trading_day_index(df):
    """
    添加交易日索引 - 这是FinRL环境所必需的

    Parameters:
    df (DataFrame): 处理后的数据

    Returns:
    df (DataFrame): 添加了交易日索引的数据
    """
    print("添加交易日索引...")

    # 确保日期格式正确
    df["date"] = pd.to_datetime(df["date"])

    # 获取唯一的日期并排序
    unique_dates = list(df["date"].unique())
    unique_dates.sort()

    # 创建日期到索引的映射
    date_to_index = {date: i for i, date in enumerate(unique_dates)}

    # 添加交易日索引列
    df["day"] = df["date"].map(date_to_index)

    # 确保索引是整数
    df["day"] = df["day"].astype(int)

    # 数据验证
    print(f"添加了交易日索引，共{len(date_to_index)}个交易日")
    print(f"第一个交易日: {unique_dates[0]}, 最后一个交易日: {unique_dates[-1]}")

    # 把date转回字符串格式以便保存到CSV
    df["date"] = df["date"].astype(str)

    return df


def add_technical_indicators(df):
    """
    添加技术指标

    Parameters:
    df (DataFrame): 原始股票数据

    Returns:
    processed (DataFrame): 添加了技术指标的数据
    """
    print("添加技术指标...")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    # 显示添加了哪些特征
    added_features = [col for col in processed.columns if col not in df.columns]
    print(f"添加了{len(added_features)}个技术指标: {', '.join(added_features[:5])}...")

    return processed


def split_data(processed_full, save=True):
    """
    将数据分割为训练集、测试集和交易集

    Parameters:
    processed_full (DataFrame): 处理后的完整数据
    save (bool): 是否保存到文件

    Returns:
    tuple: (train, test, trade) 数据集
    """
    print("分割数据集...")

    # 使用配置文件中的日期范围进行分割
    train = data_split(
        processed_full, config.DEMO_TRAIN_START_DATE, config.DEMO_TRAIN_END_DATE
    )
    test = data_split(
        processed_full, config.DEMO_TEST_START_DATE, config.DEMO_TEST_END_DATE
    )
    trade = data_split(
        processed_full, config.DEMO_TRADE_START_DATE, config.DEMO_TRADE_END_DATE
    )

    print(f"训练集: {len(train)}行，从{train['date'].min()}到{train['date'].max()}")
    print(f"测试集: {len(test)}行，从{test['date'].min()}到{test['date'].max()}")
    print(f"交易集: {len(trade)}行，从{trade['date'].min()}到{trade['date'].max()}")

    if save:
        # 保存拆分后的数据集
        train.to_csv(f"{config.DATA_SAVE_DIR}/train.csv", index=True)
        test.to_csv(f"{config.DATA_SAVE_DIR}/test.csv", index=True)
        trade.to_csv(f"{config.DATA_SAVE_DIR}/trade.csv", index=True)

        # 保存处理后的完整数据
        processed_full.to_csv(f"{config.DATA_SAVE_DIR}/processed_full.csv", index=True)

        print(f"数据集已保存到{config.DATA_SAVE_DIR}目录")

    return train, test, trade


def process_all_data(ticker_type="dow30"):
    """
    处理指定指数的全部数据，添加技术指标并进行数据集拆分

    Parameters:
    ticker_type (str): 指数类型，可选 "dow30", "sp500", "nasdaq100" 或 "all"

    Returns:
    tuple: (processed_full, train, test, trade) 数据
    """
    # 确保数据目录存在
    if not os.path.exists(config.DATA_SAVE_DIR):
        os.makedirs(config.DATA_SAVE_DIR)

    # 1. 加载原始数据
    df = load_data(ticker_type)

    # 2. 先添加技术指标 (在原始数据上计算更准确)
    processed = add_technical_indicators(df)

    # 3. 确保数据连续性 (填充缺失值)
    processed_full = ensure_data_continuity(processed)

    # 4. 添加交易日索引 (作为最后的环境准备步骤)
    processed_full = add_trading_day_index(processed_full)

    # 5. 拆分数据集
    train, test, trade = split_data(processed_full)
    return processed_full, train, test, trade


def main():
    parser = argparse.ArgumentParser(description="添加技术指标并划分训练集")
    parser.add_argument(
        "--ticker_type",
        type=str,
        default="dow30",
        choices=["dow30", "sp500", "nasdaq100", "all"],
        help="选择处理哪个指数的数据 (default: dow30)",
    )
    args = parser.parse_args()

    print("开始处理数据...")
    processed_full, train, test, trade = process_all_data(args.ticker_type)

    print("\n数据处理完成!")
    print(f"技术指标列表: {config.INDICATORS}")
    print("\n训练集示例:")
    print(train.head())


if __name__ == "__main__":
    main()
