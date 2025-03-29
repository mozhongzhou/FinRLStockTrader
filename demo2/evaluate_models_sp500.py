import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "SimSun",
]  # 优先使用的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
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
    env = StockTradingEnv(
        df=df, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )
    return env
    # env = StockTradingEnv(df=df, **env_kwargs)
    # return env


def load_models():
    """加载已训练好的模型"""
    models = {}

    # 检查并加载A2C模型
    a2c_path = f"{TRAINED_MODEL_DIR}/a2c_model_100000_steps.zip"
    if os.path.exists(a2c_path):
        models["a2c"] = A2C.load(a2c_path, device="cpu")
        print("已加载A2C模型")

    # 检查并加载PPO模型
    ppo_path = f"{TRAINED_MODEL_DIR}/ppo_spy_500.zip"
    if os.path.exists(ppo_path):
        models["ppo"] = PPO.load(ppo_path, device="cpu")
        print("已加载PPO模型")

    # # 检查并加载DDPG模型
    # ddpg_path = f"{TRAINED_MODEL_DIR}/"
    # if os.path.exists(ddpg_path):
    #     models["ddpg"] = DDPG.load(ddpg_path)
    #     print("已加载DDPG模型")

    # # 检查并加载TD3模型
    # td3_path = f"{TRAINED_MODEL_DIR}/"
    # if os.path.exists(td3_path):
    #     models["td3"] = TD3.load(td3_path)
    #     print("已加载TD3模型")

    # # 检查并加载SAC模型
    # sac_path = f"{TRAINED_MODEL_DIR}/"
    # if os.path.exists(sac_path):
    #     models["sac"] = SAC.load(sac_path)
    #     print("已加载SAC模型")

    return models


def plot_model_performance_comparison(trading_results, df_sp500_aligned):
    """
    绘制不同模型与基准的性能对比图表

    Args:
        trading_results (dict): 包含不同模型结果的字典
        df_sp500_aligned (DataFrame): 基准数据
    """
    # 收集所有模型和基准的统计数据
    model_names = list(trading_results.keys()) + ["S&P 500"]
    annual_returns = []
    cumulative_returns = []
    annual_volatility = []
    sharpe_ratios = []
    calmar_ratios = []
    max_drawdowns = []

    # 计算每个模型的统计数据
    for model_name in trading_results.keys():
        df_stat = trading_results[model_name].copy()
        df_stat = df_stat.reset_index()
        df_stat.columns.values[0] = "date"
        model_stats = backtest_stats(df_stat, value_col_name=model_name)

        annual_returns.append(model_stats[0])
        cumulative_returns.append(model_stats[1])
        annual_volatility.append(model_stats[2])
        sharpe_ratios.append(model_stats[3])
        calmar_ratios.append(model_stats[4])
        max_drawdowns.append(model_stats[5])

    # 添加基准统计数据
    df_sp500_stat = df_sp500_aligned.copy()
    df_sp500_stat = df_sp500_stat.reset_index()
    df_sp500_stat.columns.values[0] = "date"
    baseline_stats = backtest_stats(df_sp500_stat, value_col_name="S&P 500")

    annual_returns.append(baseline_stats[0])
    cumulative_returns.append(baseline_stats[1])
    annual_volatility.append(baseline_stats[2])
    sharpe_ratios.append(baseline_stats[3])
    calmar_ratios.append(baseline_stats[4])
    max_drawdowns.append(baseline_stats[5])

    # 创建主图和子图
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("强化学习模型性能对比分析", fontsize=24)

    # 设置颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # 子图1: 年化收益率
    ax1 = fig.add_subplot(3, 2, 1)
    bars1 = ax1.bar(
        model_names, [r * 100 for r in annual_returns], color=colors[: len(model_names)]
    )
    ax1.set_title("年化收益率 (%)", fontsize=16)
    ax1.set_ylabel("百分比 (%)")
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    # 子图2: 累计收益率
    ax2 = fig.add_subplot(3, 2, 2)
    bars2 = ax2.bar(
        model_names,
        [r * 100 for r in cumulative_returns],
        color=colors[: len(model_names)],
    )
    ax2.set_title("累计收益率 (%)", fontsize=16)
    ax2.set_ylabel("百分比 (%)")
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    # 子图3: 年化波动率
    ax3 = fig.add_subplot(3, 2, 3)
    bars3 = ax3.bar(
        model_names,
        [r * 100 for r in annual_volatility],
        color=colors[: len(model_names)],
    )
    ax3.set_title("年化波动率 (%)", fontsize=16)
    ax3.set_ylabel("百分比 (%)")
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    # 子图4: 夏普比率
    ax4 = fig.add_subplot(3, 2, 4)
    bars4 = ax4.bar(model_names, sharpe_ratios, color=colors[: len(model_names)])
    ax4.set_title("夏普比率", fontsize=16)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # 子图5: 卡玛比率
    ax5 = fig.add_subplot(3, 2, 5)
    bars5 = ax5.bar(model_names, calmar_ratios, color=colors[: len(model_names)])
    ax5.set_title("卡玛比率", fontsize=16)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # 子图6: 最大回撤
    ax6 = fig.add_subplot(3, 2, 6)
    bars6 = ax6.bar(
        model_names, [r * 100 for r in max_drawdowns], color=colors[: len(model_names)]
    )
    ax6.set_title("最大回撤 (%)", fontsize=16)
    ax6.set_ylabel("百分比 (%)")
    for bar in bars6:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    # 调整布局，避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # 保存组合图
    plt.savefig(f"{RESULTS_DIR}/model_performance_comparison.png", dpi=300)

    # 保存单独的指标图表
    metrics = [
        ("年化收益率", annual_returns, True),
        ("累计收益率", cumulative_returns, True),
        ("年化波动率", annual_volatility, True),
        ("夏普比率", sharpe_ratios, False),
        ("卡玛比率", calmar_ratios, False),
        ("最大回撤", max_drawdowns, True),
    ]

    for metric_name, metric_values, is_percentage in metrics:
        plt.figure(figsize=(10, 6))

        if is_percentage:
            bars = plt.bar(
                model_names,
                [v * 100 for v in metric_values],
                color=colors[: len(model_names)],
            )
            plt.ylabel("百分比 (%)")
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.2f}%",
                    ha="center",
                    va="bottom",
                )
        else:
            bars = plt.bar(model_names, metric_values, color=colors[: len(model_names)])
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.title(f"{metric_name}对比", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{metric_name}_comparison.png", dpi=300)
        plt.close()

    print(f"所有性能对比图表已保存到 {RESULTS_DIR} 目录")


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

    # 获取基准指数数据（标普500指数）
    print("=== 获取基准指数数据 ===")
    df_sp500 = get_baseline(ticker="^GSPC", start=TRADE_START_DATE, end=TRADE_END_DATE)

    # 调整基准数据与交易数据对齐
    df_sp500_aligned = pd.DataFrame()
    first_model = list(trading_results.keys())[0] if trading_results else None
    if first_model:
        df_sp500_aligned["date"] = trading_results[first_model].index
        initial_amount = 20000
        df_sp500_aligned["S&P 500"] = (
            df_sp500["close"] / df_sp500["close"][0] * initial_amount
        )
        df_sp500_aligned.set_index("date", inplace=True)

        # 合并所有结果
        result = pd.DataFrame()
        for model_name, df_model_result in trading_results.items():
            result = pd.merge(
                result, df_model_result, how="outer", left_index=True, right_index=True
            )

        # 添加基准结果
        result = pd.merge(
            result, df_sp500_aligned, how="outer", left_index=True, right_index=True
        )

        # 保存合并结果
        result.to_csv(f"{RESULTS_DIR}/evaluation_results.csv")

        # 绘制结果对比图 - 使用更简洁的方式，与示例代码一致
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.figure()
        result.plot()
        plt.title("投资组合价值比较")
        plt.xlabel("日期")
        plt.ylabel("价值 (元)")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/portfolio_comparison.png", dpi=300)
        plt.close()
        print("\n=== 生成模型性能对比图表 ===")
        plot_model_performance_comparison(trading_results, df_sp500_aligned)
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
