import os
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
import sys
import torch

torch.cuda.init()
torch.cuda.empty_cache()
# 添加项目根目录到路径
sys.path.append(".")

# 导入配置
from config import *

# 目录设置 - 使用配置文件中的定义，但在demo2子目录下
RESULTS_DIR = os.path.join("demo2", RESULTS_DIR)
TRAINED_MODEL_DIR = os.path.join("demo2", TRAINED_MODEL_DIR)
TENSORBOARD_LOG_DIR = os.path.join("demo2", TENSORBOARD_LOG_DIR)

# 确保目录存在
for dir_path in [RESULTS_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ============== 训练控制参数 ==============
# 设置为True表示训练该模型，设置为False表示跳过
if_using_a2c = True  # 是否训练A2C模型
if_using_ppo = True  # 是否训练PPO模型
if_using_ddpg = True  # 是否训练DDPG模型
if_using_td3 = True  # 是否训练TD3模型
if_using_sac = True  # 是否训练SAC模型

# 训练总步数
total_timesteps = TOTAL_TIME_STEPS  # 可以根据需要调整


# 随机种子，设置为None表示使用随机种子
seed = 10086  # 可以设置为特定整数
# =========================================


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


def train_models(
    train_data,
    if_using_a2c=False,
    if_using_ppo=True,
    if_using_ddpg=False,
    if_using_td3=False,
    if_using_sac=False,
    total_timesteps=50000,
    seed=None,
):
    """
    根据用户选择训练多个强化学习模型并保存

    Args:
        train_data (DataFrame): 训练数据
        if_using_a2c (bool): 是否训练A2C模型
        if_using_ppo (bool): 是否训练PPO模型
        if_using_ddpg (bool): 是否训练DDPG模型
        if_using_td3 (bool): 是否训练TD3模型
        if_using_sac (bool): 是否训练SAC模型
        total_timesteps (int): 训练总步数
        seed (int): 随机种子

    Returns:
        dict: 训练好的模型字典
    """
    # 构建训练环境
    env_train = build_environment(train_data)
    env_gym, _ = env_train.get_sb_env()

    # 存储训练好的模型
    trained_models = {}

    # 训练A2C模型
    if if_using_a2c:
        print("=== 开始训练 A2C 模型 ===")
        agent = DRLAgent(env=env_gym)
        model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS, seed=seed)

        # 设置日志
        tmp_path = f"{TENSORBOARD_LOG_DIR}/a2c"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_a2c.set_logger(new_logger)

        # 训练模型
        trained_a2c = agent.train_model(
            model=model_a2c, tb_log_name="a2c", total_timesteps=total_timesteps
        )

        # 保存模型
        trained_a2c.save(f"{TRAINED_MODEL_DIR}/a2c_spy_500")
        trained_models["a2c"] = trained_a2c
        print("A2C模型训练完成并保存")

    # 训练PPO模型
    if if_using_ppo:
        print("=== 开始训练 PPO 模型 ===")
        agent = DRLAgent(env=env_gym)
        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=seed)

        # 设置日志
        tmp_path = f"{TENSORBOARD_LOG_DIR}/ppo"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger)

        # 训练模型
        trained_ppo = agent.train_model(
            model=model_ppo, tb_log_name="ppo", total_timesteps=total_timesteps
        )

        # 保存模型
        trained_ppo.save(f"{TRAINED_MODEL_DIR}/ppo_spy_500")
        trained_models["ppo"] = trained_ppo
        print("PPO模型训练完成并保存")

    # 训练DDPG模型
    if if_using_ddpg:
        print("=== 开始训练 DDPG 模型 ===")
        agent = DRLAgent(env=env_gym)
        model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS, seed=seed)

        # 设置日志
        tmp_path = f"{TENSORBOARD_LOG_DIR}/ddpg"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ddpg.set_logger(new_logger)

        # 训练模型
        trained_ddpg = agent.train_model(
            model=model_ddpg, tb_log_name="ddpg", total_timesteps=total_timesteps
        )

        # 保存模型
        trained_ddpg.save(f"{TRAINED_MODEL_DIR}/ddpg_spy_500")
        trained_models["ddpg"] = trained_ddpg
        print("DDPG模型训练完成并保存")

    # 训练TD3模型
    if if_using_td3:
        print("=== 开始训练 TD3 模型 ===")
        agent = DRLAgent(env=env_gym)
        model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS, seed=seed)

        # 设置日志
        tmp_path = f"{TENSORBOARD_LOG_DIR}/td3"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_td3.set_logger(new_logger)

        # 训练模型
        trained_td3 = agent.train_model(
            model=model_td3, tb_log_name="td3", total_timesteps=total_timesteps
        )

        # 保存模型
        trained_td3.save(f"{TRAINED_MODEL_DIR}/td3_spy_500")
        trained_models["td3"] = trained_td3
        print("TD3模型训练完成并保存")

    # 训练SAC模型
    if if_using_sac:
        print("=== 开始训练 SAC 模型 ===")
        agent = DRLAgent(env=env_gym)
        model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS, seed=seed)

        # 设置日志
        tmp_path = f"{TENSORBOARD_LOG_DIR}/sac"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_sac.set_logger(new_logger)

        # 训练模型
        trained_sac = agent.train_model(
            model=model_sac, tb_log_name="sac", total_timesteps=total_timesteps
        )

        # 保存模型
        trained_sac.save(f"{TRAINED_MODEL_DIR}/sac_dow_30")
        trained_models["sac"] = trained_sac
        print("SAC模型训练完成并保存")

    return trained_models


def main():
    """主函数，专注于训练模型"""
    # 使用文件顶部定义的全局变量，不需要命令行参数
    global if_using_a2c, if_using_ppo, if_using_ddpg, if_using_td3, if_using_sac
    global total_timesteps, seed

    print("=== 训练模型流程开始 ===")
    print(
        f"训练日期范围: {TRAIN_START_DATE} 至 {TRAIN_END_DATE} (请注意,以提供的训练集为准,准确范围为整个训练集)"
    )
    print(
        f"训练模型: A2C={if_using_a2c}, PPO={if_using_ppo}, DDPG={if_using_ddpg}, TD3={if_using_td3}, SAC={if_using_sac}"
    )
    print(f"训练步数: {total_timesteps}, 随机种子: {seed if seed else '随机'}")

    # 使用配置中的数据目录
    train_path = os.path.join(DATA_SAVE_DIR, "train.csv")

    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")

    # 直接加载数据
    train = pd.read_csv(train_path)
    train = train.set_index(train.columns[0])
    train.index.names = [""]
    print(f"训练集大小: {len(train)} 行, {train['tic'].nunique()} 支股票")
    print(f"使用的技术指标: {INDICATORS}")

    # 训练模型
    print("=== 开始训练模型 ===")
    trained_models = train_models(
        train_data=train,
        if_using_a2c=if_using_a2c,
        if_using_ppo=if_using_ppo,
        if_using_ddpg=if_using_ddpg,
        if_using_td3=if_using_td3,
        if_using_sac=if_using_sac,
        total_timesteps=total_timesteps,
        seed=seed,
    )

    if not trained_models:
        print("警告: 没有选择要训练的模型")
    else:
        print("=== 训练完成 ===")
        print(f"所有模型已保存到 {TRAINED_MODEL_DIR} 目录")
        print("请运行 evaluate_models.py 进行模型评估")


if __name__ == "__main__":
    main()
