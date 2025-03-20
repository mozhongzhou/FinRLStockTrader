import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
import datetime
import sys

# 添加项目根目录到路径
sys.path.append(".")

# 导入配置
from config import *
from config_tickers import DOW_30_TICKER

# 配置参数 - 优先使用DEMO参数，如果不存在则使用标准参数
TRAIN_START_DATE = (
    DEMO_TRAIN_START_DATE if "DEMO_TRAIN_START_DATE" in globals() else TRAIN_START_DATE
)
TRAIN_END_DATE = (
    DEMO_TRAIN_END_DATE if "DEMO_TRAIN_END_DATE" in globals() else TRAIN_END_DATE
)
TEST_START_DATE = (
    DEMO_TEST_START_DATE if "DEMO_TEST_START_DATE" in globals() else TEST_START_DATE
)
TEST_END_DATE = (
    DEMO_TEST_END_DATE if "DEMO_TEST_END_DATE" in globals() else TEST_END_DATE
)

# 技术指标列表 - 使用配置文件中的定义
INDICATORS = INDICATORS  # 直接从config.py导入

# 目录设置 - 使用配置文件中的定义，但在demo2子目录下
RESULTS_DIR = os.path.join("demo2", RESULTS_DIR)
TRAINED_MODEL_DIR = os.path.join("demo2", TRAINED_MODEL_DIR)
TENSORBOARD_LOG_DIR = os.path.join("demo2", TENSORBOARD_LOG_DIR)

# 确保目录存在
for dir_path in [RESULTS_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def build_environment(df):
    """
    构建交易环境

    Args:
        df (DataFrame): 股票数据

    Returns:
        env: 交易环境实例
    """
    # 确定维度
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

    # 交易成本和初始持仓
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # 环境参数
    env_kwargs = {
        "hmax": 100,  # 最大交易数量
        "initial_amount": 20000,  # 初始资金
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # 训练环境
    env = StockTradingEnv(df=df, **env_kwargs)
    return env


def train_models(train_data):
    """
    训练多个强化学习模型并保存

    Args:
        train_data (DataFrame): 训练数据

    Returns:
        dict: 训练好的模型字典
    """
    # 构建训练环境
    env_train = build_environment(train_data)
    env_gym, _ = env_train.get_sb_env()

    # 初始化DRL智能体
    agent = DRLAgent(env=env_gym)

    # 存储训练好的模型
    trained_models = {}

    print("=== 开始训练 A2C 模型 ===")
    # 获取A2C模型，使用配置文件中的参数
    model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS)
    # 设置日志
    tmp_path = f"{TENSORBOARD_LOG_DIR}/a2c"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger)
    # 训练模型
    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=50000
    )
    # 保存模型
    trained_a2c.save(f"{TRAINED_MODEL_DIR}/a2c_dow_30")
    trained_models["a2c"] = trained_a2c

    print("=== 开始训练 PPO 模型 ===")
    # 获取PPO模型，使用配置文件中的参数
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    # 设置日志
    tmp_path = f"{TENSORBOARD_LOG_DIR}/ppo"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger)
    # 训练模型
    trained_ppo = agent.train_model(
        model=model_ppo, tb_log_name="ppo", total_timesteps=50000
    )
    # 保存模型
    trained_ppo.save(f"{TRAINED_MODEL_DIR}/ppo_dow_30")
    trained_models["ppo"] = trained_ppo

    print("=== 开始训练 DDPG 模型 ===")
    # 获取DDPG模型，使用配置文件中的参数
    model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
    # 设置日志
    tmp_path = f"{TENSORBOARD_LOG_DIR}/ddpg"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ddpg.set_logger(new_logger)
    # 训练模型
    trained_ddpg = agent.train_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
    )
    # 保存模型
    trained_ddpg.save(f"{TRAINED_MODEL_DIR}/ddpg_dow_30")
    trained_models["ddpg"] = trained_ddpg

    return trained_models


def main():
    """主函数，专注于训练模型"""
    print("=== 训练模型流程开始 ===")
    print(f"训练日期范围: {TRAIN_START_DATE} 至 {TRAIN_END_DATE}")
    print(f"测试日期范围: {TEST_START_DATE} 至 {TEST_END_DATE}")

    # 使用配置中的数据目录
    train_path = os.path.join(DATA_SAVE_DIR, "train.csv")
    test_path = os.path.join(DATA_SAVE_DIR, "test.csv")

    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集文件不存在: {test_path}")

    # 直接加载数据
    train = pd.read_csv(train_path)
    train = train.set_index(train.columns[0])
    train.index.names = [""]
    test = pd.read_csv(test_path)
    test = test.set_index(test.columns[0])
    test.index.names = [""]

    print(f"训练集大小: {len(train)} 行, {train['tic'].nunique()} 支股票")
    print(f"测试集大小: {len(test)} 行, {test['tic'].nunique()} 支股票")
    print(f"使用的技术指标: {INDICATORS}")

    # 训练模型
    print("=== 开始训练模型 ===")
    trained_models = train_models(train)

    print("=== 训练完成 ===")
    print(f"所有模型已保存到 {TRAINED_MODEL_DIR} 目录")
    print("请运行 evaluate_models.py 进行模型评估")


if __name__ == "__main__":
    main()
