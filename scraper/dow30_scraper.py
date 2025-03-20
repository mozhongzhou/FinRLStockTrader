import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


# 获取Dow 30股票代码
def get_dow30_tickers():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 发送请求获取网页
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"无法访问网页，状态码: {response.status_code}")

    # 解析HTML
    soup = BeautifulSoup(response.content, "html.parser")

    # 找到包含成分股的表格
    table = soup.find("table", {"class": "wikitable sortable"})
    if not table:
        raise Exception("未找到表格，请检查页面结构")

    # 提取股票代码
    tickers = []
    rows = table.find_all("tr")[1:]  # 跳过表头
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 1:
            ticker = cells[1].get_text(strip=True)  # 股票代码在第二列
            tickers.append(ticker)

    # 检查是否抓取到30个股票代码
    if len(tickers) != 30:
        print(f"警告：抓取到的股票代码数量为 {len(tickers)}，请检查数据")

    # 变更：返回DataFrame而不是列表
    return pd.DataFrame({"Ticker": tickers})


# 保存到CSV文件
def save_to_csv(tickers_df, output_dir="scraper/tickers", filename="dow30_tickers.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, filename)
    tickers_df.to_csv(csv_path, index=False)
    print(f"道指30股票代码已保存到: {csv_path}")


# 主程序
if __name__ == "__main__":
    try:
        dow30_df = get_dow30_tickers()
        save_to_csv(dow30_df)
    except Exception as e:
        print(f"发生错误: {e}")
