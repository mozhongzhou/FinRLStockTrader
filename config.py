# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2000-01-01"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2022-12-31"
# TEST_START_DATE = "2020-01-01"
# TEST_END_DATE = "2022-12-31"
TRADE_START_DATE = "2023-01-01"
TRADE_END_DATE = "2025-03-19"

DEMO_TRAIN_START_DATE = "2020-01-01"
DEMO_TRAIN_END_DATE = "2023-12-31"
# DEMO_TEST_START_DATE = "2023-01-01"
# DEMO_TEST_END_DATE = "2023-12-31"
DEMO_TRADE_START_DATE = "2024-01-01"
DEMO_TRADE_END_DATE = "2025-01-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

TOTAL_TIME_STEPS = 200000
# Model Parameters
A2C_PARAMS = {
    "n_steps": 256,  # 增加到256减少更新频率
    "ent_coef": 0.01,  # 保持探索平衡
    "learning_rate": 0.002,  # 稍微提高学习率加速收敛
    "device": "cuda",
    "normalize_advantage": True,
    "gae_lambda": 0.95,
    "gamma": 0.99,  # 添加折扣因子
    "vf_coef": 0.5,  # 价值函数系数
}

PPO_PARAMS = {
    "n_steps": 4096,  # 增加步数减少更新频率
    "batch_size": 1024,  # 大幅提高批处理大小
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "device": "cuda",
    "n_epochs": 5,  # 限制每批数据训练次数
    "clip_range": 0.2,  # 明确设置裁剪范围
}
DDPG_PARAMS = {
    "batch_size": 512,  # 增加批处理大小
    "buffer_size": 300000,  # 增加缓冲区大小
    "learning_rate": 0.001,
    "device": "cuda",
    "train_freq": (10, "step"),  # 减少训练频率
    "gradient_steps": 1,  # 限制每次更新的梯度步数
}
TD3_PARAMS = {
    "batch_size": 512,  # 大幅增加批处理大小
    "buffer_size": 500000,  # 减少内存压力但保持足够大
    "learning_rate": 0.001,
    "device": "cuda",
    "train_freq": (10, "step"),  # 减少训练频率
    "gradient_steps": 1,  # 限制梯度步数
}
SAC_PARAMS = {
    "batch_size": 512,  # 增加批处理大小
    "buffer_size": 300000,  # 适当增加缓冲区
    "learning_rate": 0.0001,
    "learning_starts": 10000,  # 大幅增加预热样本数
    "ent_coef": "auto_0.1",
    "device": "cuda",
    "train_freq": (10, "step"),  # 减少训练频率
    "gradient_steps": 1,  # 限制梯度步数
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
