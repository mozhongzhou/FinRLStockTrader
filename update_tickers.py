import os
import sys
import re
from datetime import datetime

# 添加当前目录到搜索路径，以便导入其他模块
sys.path.append(".")
sys.path.append("./scraper")

# 导入爬虫模块
from scraper.dow30_scraper import get_dow30_tickers
from scraper.sp500_scraper import get_sp500_tickers
from scraper.nasdaq100_scraper import get_nasdaq100_tickers


def update_config_tickers():
    """
    获取最新的道指30、标普500和纳斯达克100成分股，更新到config_tickers.py中
    """
    print("开始更新股票代码配置文件...")

    # 1. 获取最新的成分股数据
    print("\n=== 获取最新成分股数据 ===")
    dow30_df = get_dow30_tickers()
    sp500_df = get_sp500_tickers()
    nasdaq100_df = get_nasdaq100_tickers()

    # 2. 提取股票代码列表
    # 确保我们获取的是正确的列 - 应该是"Ticker"列，而不是其他列
    dow30_tickers = dow30_df["Ticker"].tolist() if "Ticker" in dow30_df.columns else []
    sp500_tickers = sp500_df["Ticker"].tolist() if "Ticker" in sp500_df.columns else []
    nasdaq100_tickers = (
        nasdaq100_df["Ticker"].tolist() if "Ticker" in nasdaq100_df.columns else []
    )

    # 调试输出，验证获取的股票代码正确
    print(f"道指30样例: {dow30_tickers[:5]}")
    print(f"标普500样例: {sp500_tickers[:5]}")
    print(f"纳斯达克100样例: {nasdaq100_tickers[:5]}")

    # 3. 读取当前配置文件
    config_path = "config_tickers.py"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在！")
        return False

    # 使用UTF-8编码打开文件
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()

    # 4. 备份原始配置文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"config_tickers_backup_{timestamp}.py"
    # 使用UTF-8编码保存备份
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(config_content)
        print(f"已备份原始配置文件到: {backup_path}")

    # 5. 更新DOW_30_TICKER列表
    dow_replacement = (
        'DOW_30_TICKER = [\n    "' + '",\n    "'.join(dow30_tickers) + '"\n]'
    )
    # 使用更精确的模式匹配整个DOW_30_TICKER数组定义
    dow_pattern = r'DOW_30_TICKER\s*=\s*\[\s*(?:"[^"]*"\s*,\s*)*(?:"[^"]*"\s*)*\]'
    config_content = re.sub(dow_pattern, dow_replacement, config_content)

    # 6. 更新SP_500_TICKER列表，如果它存在
    if sp500_tickers:
        sp_replacement = (
            'SP_500_TICKER = [\n    "' + '",\n    "'.join(sp500_tickers) + '"\n]'
        )
        sp_pattern = r'SP_500_TICKER\s*=\s*\[\s*(?:"[^"]*"\s*,\s*)*(?:"[^"]*"\s*)*\]'
        config_content = re.sub(sp_pattern, sp_replacement, config_content)

    # 7. 更新NAS_100_TICKER列表，如果它存在
    if nasdaq100_tickers:
        nas_replacement = (
            'NAS_100_TICKER = [\n    "' + '",\n    "'.join(nasdaq100_tickers) + '"\n]'
        )
        nas_pattern = r'NAS_100_TICKER\s*=\s*\[\s*(?:"[^"]*"\s*,\s*)*(?:"[^"]*"\s*)*\]'
        config_content = re.sub(nas_pattern, nas_replacement, config_content)

    # 8. 添加更新时间注释
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_content = re.sub(
        r"# 截至\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
        f"# 截至{update_time}",
        config_content,
    )

    # 9. 写回更新后的配置文件
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"\n成功更新配置文件: {config_path}")
    print(f"更新内容:")
    print(f"- 道指30成分股: {len(dow30_tickers)}支")
    print(f"- 标普500成分股: {len(sp500_tickers)}支")
    print(f"- 纳斯达克100成分股: {len(nasdaq100_tickers)}支")

    return True


if __name__ == "__main__":
    update_config_tickers()
