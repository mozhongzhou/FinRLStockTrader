# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2018-01-01"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2020-12-31"
# TEST_START_DATE = "2020-01-01"
# TEST_END_DATE = "2022-12-31"
TRADE_START_DATE = "2023-01-01"
TRADE_END_DATE = "2023-12-31"

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

TOTAL_TIME_STEPS = 5000
# Model Parameters
A2C_PARAMS = {
    "n_steps": 5,
    "ent_coef": 0.01,
    "learning_rate": 0.0007,
    "device": "cpu",
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
    "device": "cpu",
}
DDPG_PARAMS = {
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_rate": 0.001,
    "device": "cpu",
}
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
    "device": "cpu",
}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "device": "cpu",
}
# PPO_PARAMS = {
#     "n_steps": 2048,  # 从4096减少到2048以降低内存压力
#     "batch_size": 512,  # 从1024减少到512以降低内存使用
#     "ent_coef": 0.01,
#     "learning_rate": 0.0003,
#     "device": "cpu",
#     "n_epochs": 4,  # 从5减少到4以加快训练
#     "clip_range": 0.2,
# }
# DDPG_PARAMS = {
#     "batch_size": 256,  # 从512减小到256
#     "buffer_size": 100000,  # 从300000显著减少到100000
#     "learning_rate": 0.001,
#     "device": "cuda",
#     "train_freq": (10, "step"),
#     "gradient_steps": 1,
# }
# TD3_PARAMS = {
#     "batch_size": 256,  # 从512减小到256
#     "buffer_size": 100000,  # 从500000大幅减少到100000
#     "learning_rate": 0.001,
#     "device": "cuda",
#     "train_freq": (20, "step"),  # 从10增加到20以减少更新频率
#     "gradient_steps": 1,
# }
# SAC_PARAMS = {
#     "batch_size": 256,  # 从512减小到256
#     "buffer_size": 100000,  # 从300000减少到100000
#     "learning_rate": 0.0003,  # 从0.0001增加到0.0003以加快收敛
#     "learning_starts": 5000,  # 从10000减少到5000以加快初始阶段
#     "ent_coef": "auto_0.1",
#     "device": "cuda",
#     "train_freq": (20, "step"),  # 从10增加到20以减少更新频率
#     "gradient_steps": 1,
# }
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
