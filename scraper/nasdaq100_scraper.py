import requests
import pandas as pd
from bs4 import BeautifulSoup
import os


def get_nasdaq100_tickers():
    """
    从维基百科获取纳斯达克100指数的成分股
    """
    print("正在获取纳斯达克100成分股...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # 找到包含成分股的表格
    tables = soup.find_all("table", {"class": "wikitable"})
    target_table = None

    for table in tables:
        headers = [th.text.strip() for th in table.find_all("th")]
        if "Ticker" in headers and "Company" in headers:
            target_table = table
            break

    if not target_table:
        raise Exception("在维基百科页面上找不到纳斯达克100成分股表格")

    # 解析表格数据
    tickers = []
    company_names = []
    sectors = []

    # 确定列索引
    headers = [th.text.strip() for th in target_table.find_all("th")]
    ticker_idx = headers.index("Ticker") if "Ticker" in headers else 0
    company_idx = headers.index("Company") if "Company" in headers else 1
    sector_idx = headers.index("GICS Sector") if "GICS Sector" in headers else 2

    for row in target_table.find_all("tr")[1:]:  # 跳过表头行
        cells = row.find_all("td")
        if len(cells) > max(ticker_idx, company_idx):
            ticker = cells[ticker_idx].text.strip()
            company = cells[company_idx].text.strip()
            sector = (
                cells[sector_idx].text.strip() if len(cells) > sector_idx else "N/A"
            )

            tickers.append(ticker)
            company_names.append(company)
            sectors.append(sector)

    # 创建DataFrame
    nasdaq100_df = pd.DataFrame(
        {"Ticker": tickers, "Company": company_names, "Sector": sectors}
    )

    print(f"共获取 {len(nasdaq100_df)} 支纳斯达克100成分股")

    # 保存为CSV
    output_dir = "scraper/tickers"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "nasdaq100_tickers.csv")
    nasdaq100_df.to_csv(output_path, index=False)
    print(f"纳斯达克100成分股数据已保存到: {output_path}")

    return nasdaq100_df


if __name__ == "__main__":
    nasdaq100_df = get_nasdaq100_tickers()
    print("\n前5支纳斯达克100成分股:")
    print(nasdaq100_df.head())
