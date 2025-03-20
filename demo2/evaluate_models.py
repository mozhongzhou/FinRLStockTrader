import os
import numpy as np

# 添加NumPy兼容性补丁
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
    print("已添加NumPy兼容性补丁: np.NINF = -np.inf")
import pandas as pd
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
import datetime
import sys

# 添加项目根目录到路径
sys.path.append(".")

# 导入配置
from config import *
from config_tickers import DOW_30_TICKER

# 配置参数 - 优先使用DEMO参数，如果不存在则使用标准参数
TRADE_START_DATE = (
    DEMO_TRADE_START_DATE if "DEMO_TRADE_START_DATE" in globals() else TRADE_START_DATE
)
TRADE_END_DATE = (
    DEMO_TRADE_END_DATE if "DEMO_TRADE_END_DATE" in globals() else TRADE_END_DATE
)

# 技术指标列表 - 使用配置文件中的定义
INDICATORS = INDICATORS  # 直接从config.py导入

# 目录设置 - 使用配置文件中的定义，但在demo2子目录下
RESULTS_DIR = os.path.join("demo2", RESULTS_DIR)
TRAINED_MODEL_DIR = os.path.join("demo2", TRAINED_MODEL_DIR)

# 确保目录存在
for dir_path in [RESULTS_DIR, TRAINED_MODEL_DIR]:
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

    # 交易环境 - 设置turbulence_threshold用于风险控制
    # env = StockTradingEnv(
    #     df=df, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    # )
    # return env
    env = StockTradingEnv(df=df, **env_kwargs)
    return env


def load_models():
    """加载已训练好的模型"""
    models = {}

    # 检查并加载A2C模型
    a2c_path = f"{TRAINED_MODEL_DIR}/a2c_dow_30.zip"
    if os.path.exists(a2c_path):
        models["a2c"] = A2C.load(a2c_path)
        print("已加载A2C模型")

    # 检查并加载PPO模型
    ppo_path = f"{TRAINED_MODEL_DIR}/ppo_dow_30.zip"
    if os.path.exists(ppo_path):
        models["ppo"] = PPO.load(ppo_path)
        print("已加载PPO模型")

    # 检查并加载DDPG模型
    ddpg_path = f"{TRAINED_MODEL_DIR}/ddpg_dow_30.zip"
    if os.path.exists(ddpg_path):
        models["ddpg"] = DDPG.load(ddpg_path)
        print("已加载DDPG模型")

    return models


def evaluate_models(trade_data, models):
    """评估模型在交易数据上的表现"""

    # 创建交易环境
    trade_env = build_environment(trade_data)

    # 存储每个模型的交易结果
    trading_results = {}

    # 使用每个模型进行交易
    for model_name, model in models.items():
        print(f"=== 使用 {model_name.upper()} 模型进行评估 ===")
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=model, environment=trade_env
        )

        # 保存交易结果
        df_account_value.to_csv(
            f"{RESULTS_DIR}/{model_name}_account_value.csv", index=False
        )
        df_actions.to_csv(f"{RESULTS_DIR}/{model_name}_actions.csv", index=False)

        # 处理结果用于可视化
        df_result = df_account_value.set_index(df_account_value.columns[0])
        df_result.rename(columns={"account_value": model_name}, inplace=True)

        trading_results[model_name] = df_result

    # 获取基准指数数据（道琼斯工业平均指数）
    print("=== 获取基准指数数据 ===")
    df_dji = get_baseline(
        ticker="^DJI", start=DEMO_TRADE_START_DATE, end=DEMO_TRADE_END_DATE
    )

    # 调整基准数据与交易数据对齐
    df_dji_aligned = pd.DataFrame()
    first_model = list(trading_results.keys())[0] if trading_results else None
    if first_model:
        df_dji_aligned["date"] = trading_results[first_model].index
        initial_amount = 20000
        df_dji_aligned["dji"] = df_dji["close"] / df_dji["close"][0] * initial_amount
        df_dji_aligned.set_index("date", inplace=True)

        # 合并所有结果
        result = pd.DataFrame()
        for model_name, df_model_result in trading_results.items():
            result = pd.merge(
                result, df_model_result, how="outer", left_index=True, right_index=True
            )

        # 添加基准结果
        result = pd.merge(
            result, df_dji_aligned, how="outer", left_index=True, right_index=True
        )

        # 保存合并结果
        result.to_csv(f"{RESULTS_DIR}/evaluation_results.csv")

        # 绘制结果对比图
        plt.figure(figsize=(50, 10))
        plt.title("Portfolio Value Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        result.plot()
        plt.tight_layout()  # 确保标签完全显示
        plt.savefig(f"{RESULTS_DIR}/evaluation_comparison_plot.png", dpi=800)

        # 计算每个模型的表现统计数据
        print("\n=== 模型表现统计数据 ===")
        for model_name in trading_results.keys():
            # 准备用于统计的数据框 - 复制一份避免影响原始数据
            df_stat = trading_results[model_name].copy()
            # 重置索引，将日期变为常规列
            df_stat = df_stat.reset_index()
            # 确保第一列名为'date'，这是backtest_stats函数所期望的
            df_stat.columns.values[0] = "date"

            # 计算统计数据
            model_stats = backtest_stats(df_stat, value_col_name=model_name)

            print(f"\n{model_name.upper()} 模型:")
            print(f"年化收益率: {model_stats[0]:.2%}")
            print(f"累计收益率: {model_stats[1]:.2%}")
            print(f"年化波动率: {model_stats[2]:.2%}")
            print(f"夏普比率: {model_stats[3]:.2f}")
            print(f"卡玛比率: {model_stats[4]:.2f}")
            print(f"最大回撤: {model_stats[5]:.2%}")

        # 基准统计也需要同样处理
        df_dji_stat = df_dji_aligned.copy()
        df_dji_stat = df_dji_stat.reset_index()
        df_dji_stat.columns.values[0] = "date"
        baseline_stats = backtest_stats(df_dji_stat, value_col_name="dji")

        print("\n基准 (DJI):")
        print(f"年化收益率: {baseline_stats[0]:.2%}")
        print(f"累计收益率: {baseline_stats[1]:.2%}")
        print(f"年化波动率: {baseline_stats[2]:.2%}")
        print(f"夏普比率: {baseline_stats[3]:.2f}")
        print(f"卡玛比率: {baseline_stats[4]:.2f}")
        print(f"最大回撤: {baseline_stats[5]:.2%}")

        return result

    print("警告: 没有找到交易结果")
    return None


def main():
    """主函数，专注于评估模型性能"""
    print("=== 评估模型流程开始 ===")
    print(f"交易日期范围: {TRADE_START_DATE} 至 {TRADE_END_DATE}")

    trade_path = os.path.join(DATA_SAVE_DIR, "trade.csv")
    # 检查文件是否存在
    if not os.path.exists(trade_path):
        raise FileNotFoundError(f"测试集文件不存在: {trade_path}")

    # 直接加载数据
    trade = pd.read_csv(trade_path)
    trade = trade.set_index(trade.columns[0])
    trade.index.names = [""]
    # 加载模型
    print("=== 加载训练好的模型 ===")
    models = load_models()

    if not models:
        print("错误: 未找到任何训练好的模型，请先运行train_agent.py")
        return

    # 评估模型
    print("=== 开始评估模型 ===")
    results = evaluate_models(trade, models)

    print("=== 评估完成 ===")
    print(f"所有结果已保存到 {RESULTS_DIR} 目录")


if __name__ == "__main__":
    main()
