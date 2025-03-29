import os
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# 导入检查点回调功能
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import sys
import torch
import time

# 为了IDE智能提示和跳转，可以添加以下导入
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC

# CUDA初始化
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
CHECKPOINT_DIR = os.path.join("demo2", "checkpoints")  # 新增检查点目录

# 确保目录存在
for dir_path in [RESULTS_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, CHECKPOINT_DIR]:
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

# 检查点保存频率 (步数)
checkpoint_freq = 20000  # 每20000步保存一次检查点

# 随机种子，设置为None表示使用随机种子
seed = 10086  # 可以设置为特定整数
# =========================================


# 自定义回调函数，显示训练进度和时间统计
class ProgressCallback(BaseCallback):
    def __init__(self, model_name, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.model_name = model_name
        self.start_time = None
        self.last_time = None
        self.last_timestep = 0

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        print(f"开始训练 {self.model_name} 模型...")

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            now = time.time()
            steps_done = self.n_calls - self.last_timestep
            time_taken = now - self.last_time
            steps_per_second = steps_done / time_taken if time_taken > 0 else 0

            # 计算总进度和预估剩余时间
            progress = self.n_calls / self.model.num_timesteps
            elapsed_time = now - self.start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time

            # 转换为时分秒格式
            remaining_hours = int(remaining_time // 3600)
            remaining_minutes = int((remaining_time % 3600) // 60)
            remaining_seconds = int(remaining_time % 60)

            print(
                f"{self.model_name} 训练进度: {self.n_calls}/{self.model.num_timesteps} "
                f"({progress:.1%}) - 速度: {steps_per_second:.1f}步/秒 - "
                f"剩余时间: {remaining_hours}小时{remaining_minutes}分钟{remaining_seconds}秒"
            )

            self.last_time = now
            self.last_timestep = self.n_calls
        return True

    def _on_training_end(self):
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(
            f"{self.model_name} 模型训练完成！总用时: {hours}小时{minutes}分钟{seconds}秒"
        )


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
        "initial_amount": 1000000,  # 初始资金
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
    total_timesteps=5000,
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

    # 训练PPO模型
    if if_using_ppo:
        try:
            print("=== 开始训练 PPO 模型 ===")
            # 创建检查点目录
            ppo_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "ppo")
            if not os.path.exists(ppo_checkpoint_dir):
                os.makedirs(ppo_checkpoint_dir)

            # 检查是否有现有检查点
            latest_checkpoint = find_latest_checkpoint(ppo_checkpoint_dir)

            agent = DRLAgent(env=env_gym)

            if latest_checkpoint:
                print(f"发现PPO检查点: {latest_checkpoint}，从该检查点继续训练")
                model_ppo = agent.get_model("ppo", model_path=latest_checkpoint)
            else:
                model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=seed)

            # 设置日志
            tmp_path = f"{TENSORBOARD_LOG_DIR}/ppo"
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model_ppo.set_logger(new_logger)

            # 创建检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=ppo_checkpoint_dir,
                name_prefix="ppo_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            # 创建进度回调
            progress_callback = ProgressCallback(model_name="PPO")

            # # 训练模型
            # trained_ppo = agent.train_model(
            #     model=model_ppo,
            #     tb_log_name="ppo",
            #     total_timesteps=total_timesteps,
            #     callback=[checkpoint_callback, progress_callback],
            # ) 有问题  callback是硬编码 不是可选的一个参数  不能添加 要换一个思路 换一个函数方法
            trained_ppo = model_ppo.learn(
                total_timesteps=total_timesteps,
                tb_log_name="ppo",
                callback=[checkpoint_callback, progress_callback],
            )

            # 保存最终模型
            trained_ppo.save(f"{TRAINED_MODEL_DIR}/ppo_spy_500")
            trained_models["ppo"] = trained_ppo
            print("PPO模型训练完成并保存")

            # 清理内存
            del trained_ppo
            del model_ppo
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"PPO模型训练出错: {str(e)}")

    # 训练A2C模型
    if if_using_a2c:
        try:
            print("=== 开始训练 A2C 模型 ===")
            # 创建检查点目录
            a2c_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "a2c")
            if not os.path.exists(a2c_checkpoint_dir):
                os.makedirs(a2c_checkpoint_dir)

            # 检查是否有现有检查点可以继续训练
            latest_checkpoint = find_latest_checkpoint(a2c_checkpoint_dir)

            agent = DRLAgent(env=env_gym)

            if latest_checkpoint:
                print(f"发现A2C检查点: {latest_checkpoint}，从该检查点继续训练")
                model_a2c = agent.get_model("a2c", model_path=latest_checkpoint)
            else:
                model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS, seed=seed)

            # 设置日志
            tmp_path = f"{TENSORBOARD_LOG_DIR}/a2c"
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model_a2c.set_logger(new_logger)

            # 创建检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=a2c_checkpoint_dir,
                name_prefix="a2c_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            # 创建进度回调
            progress_callback = ProgressCallback(model_name="A2C")

            # 训练模型
            trained_a2c = model_a2c.learn(
                total_timesteps=total_timesteps,
                tb_log_name="a2c",
                callback=[checkpoint_callback, progress_callback],
            )

            # 保存最终模型
            trained_a2c.save(f"{TRAINED_MODEL_DIR}/a2c_spy_500")
            trained_models["a2c"] = trained_a2c
            print("A2C模型训练完成并保存")

            # 清理内存
            del trained_a2c
            del model_a2c
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"A2C模型训练出错: {str(e)}")

    # 训练DDPG模型
    if if_using_ddpg:
        try:
            print("=== 开始训练 DDPG 模型 ===")
            # 创建检查点目录
            ddpg_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "ddpg")
            if not os.path.exists(ddpg_checkpoint_dir):
                os.makedirs(ddpg_checkpoint_dir)

            # 检查是否有现有检查点
            latest_checkpoint = find_latest_checkpoint(ddpg_checkpoint_dir)

            agent = DRLAgent(env=env_gym)

            if latest_checkpoint:
                print(f"发现DDPG检查点: {latest_checkpoint}，从该检查点继续训练")
                model_ddpg = agent.get_model("ddpg", model_path=latest_checkpoint)
            else:
                model_ddpg = agent.get_model(
                    "ddpg", model_kwargs=DDPG_PARAMS, seed=seed
                )

            # 设置日志
            tmp_path = f"{TENSORBOARD_LOG_DIR}/ddpg"
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model_ddpg.set_logger(new_logger)

            # 创建检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=ddpg_checkpoint_dir,
                name_prefix="ddpg_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            # 创建进度回调
            progress_callback = ProgressCallback(model_name="DDPG")

            # 训练模型
            trained_ddpg = model_ddpg.learn(
                total_timesteps=total_timesteps,
                tb_log_name="ddpg",
                callback=[checkpoint_callback, progress_callback],
            )

            # 保存最终模型
            trained_ddpg.save(f"{TRAINED_MODEL_DIR}/ddpg_spy_500")
            trained_models["ddpg"] = trained_ddpg
            print("DDPG模型训练完成并保存")

            # 清理内存
            del trained_ddpg
            del model_ddpg
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"DDPG模型训练出错: {str(e)}")

    # 训练TD3模型
    if if_using_td3:
        try:
            print("=== 开始训练 TD3 模型 ===")
            # 创建检查点目录
            td3_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "td3")
            if not os.path.exists(td3_checkpoint_dir):
                os.makedirs(td3_checkpoint_dir)

            # 检查是否有现有检查点
            latest_checkpoint = find_latest_checkpoint(td3_checkpoint_dir)

            agent = DRLAgent(env=env_gym)

            if latest_checkpoint:
                print(f"发现TD3检查点: {latest_checkpoint}，从该检查点继续训练")
                model_td3 = agent.get_model("td3", model_path=latest_checkpoint)
            else:
                model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS, seed=seed)

            # 设置日志
            tmp_path = f"{TENSORBOARD_LOG_DIR}/td3"
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model_td3.set_logger(new_logger)

            # 创建检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=td3_checkpoint_dir,
                name_prefix="td3_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            # 创建进度回调
            progress_callback = ProgressCallback(model_name="TD3")

            # 训练模型
            trained_td3 = model_td3.learn(
                total_timesteps=total_timesteps,
                tb_log_name="td3",
                callback=[checkpoint_callback, progress_callback],
            )

            # 保存最终模型
            trained_td3.save(f"{TRAINED_MODEL_DIR}/td3_spy_500")
            trained_models["td3"] = trained_td3
            print("TD3模型训练完成并保存")

            # 清理内存
            del trained_td3
            del model_td3
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"TD3模型训练出错: {str(e)}")

    # 训练SAC模型
    if if_using_sac:
        try:
            print("=== 开始训练 SAC 模型 ===")
            # 创建检查点目录
            sac_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "sac")
            if not os.path.exists(sac_checkpoint_dir):
                os.makedirs(sac_checkpoint_dir)

            # 检查是否有现有检查点
            latest_checkpoint = find_latest_checkpoint(sac_checkpoint_dir)

            agent = DRLAgent(env=env_gym)

            if latest_checkpoint:
                print(f"发现SAC检查点: {latest_checkpoint}，从该检查点继续训练")
                model_sac = agent.get_model("sac", model_path=latest_checkpoint)
            else:
                model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS, seed=seed)

            # 设置日志
            tmp_path = f"{TENSORBOARD_LOG_DIR}/sac"
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model_sac.set_logger(new_logger)

            # 创建检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=sac_checkpoint_dir,
                name_prefix="sac_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            # 创建进度回调
            progress_callback = ProgressCallback(model_name="SAC")

            # 训练模型
            trained_sac = model_sac.learn(
                total_timesteps=total_timesteps,
                tb_log_name="sac",
                callback=[checkpoint_callback, progress_callback],
            )

            # 保存最终模型
            trained_sac.save(f"{TRAINED_MODEL_DIR}/sac_dow_30")
            trained_models["sac"] = trained_sac
            print("SAC模型训练完成并保存")

            # 清理内存
            del trained_sac
            del model_sac
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"SAC模型训练出错: {str(e)}")

    return trained_models


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not checkpoint_files:
        return None

    # 按文件修改时间排序
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True
    )
    return os.path.join(checkpoint_dir, checkpoint_files[0])


def main():
    """主函数，专注于训练模型"""
    # 使用文件顶部定义的全局变量，不需要命令行参数
    global if_using_a2c, if_using_ppo, if_using_ddpg, if_using_td3, if_using_sac
    global total_timesteps, seed, checkpoint_freq

    print("=== 训练模型流程开始 ===")
    print(
        f"训练日期范围: {TRAIN_START_DATE} 至 {TRAIN_END_DATE} (请注意,以提供的训练集为准,准确范围为整个训练集)"
    )
    print(
        f"训练模型: A2C={if_using_a2c}, PPO={if_using_ppo}, DDPG={if_using_ddpg}, TD3={if_using_td3}, SAC={if_using_sac}"
    )
    print(f"训练步数: {total_timesteps}, 随机种子: {seed if seed else '随机'}")
    print(f"检查点保存频率: 每{checkpoint_freq}步")
    print(f"检查点保存目录: {CHECKPOINT_DIR}")

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
    try:
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
    except KeyboardInterrupt:
        print("\n训练被用户中断！已保存的检查点位于 " + CHECKPOINT_DIR)
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        print("您可以从最新的检查点恢复训练")


if __name__ == "__main__":
    main()
