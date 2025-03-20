import requests
import pandas as pd
from bs4 import BeautifulSoup
import os


def get_sp500_tickers():
    """
    从维基百科获取标普500指数的成分股
    """
    print("正在获取标普500成分股...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # 找到包含成分股的表格
    table = soup.find("table", {"id": "constituents"})

    if not table:
        raise Exception("在维基百科页面上找不到标普500成分股表格")

    # 解析表格数据
    tickers = []
    company_names = []
    sectors = []

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) >= 3:
            ticker = cells[0].text.strip()
            company = cells[1].text.strip()
            sector = cells[2].text.strip()

            # 转换股票代码格式 (将 ticker.x 转换为 ticker-x)
            ticker = ticker.replace(".", "-")

            tickers.append(ticker)
            company_names.append(company)
            sectors.append(sector)

    # 创建DataFrame
    sp500_df = pd.DataFrame(
        {"Ticker": tickers, "Company": company_names, "Sector": sectors}
    )

    print(f"共获取 {len(sp500_df)} 支标普500成分股")

    # 保存为CSV
    output_dir = "scraper/tickers"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "sp500_tickers.csv")
    sp500_df.to_csv(output_path, index=False)
    print(f"标普500成分股数据已保存到: {output_path}")

    return sp500_df


if __name__ == "__main__":
    sp500_df = get_sp500_tickers()
    print("\n前5支标普500成分股:")
    print(sp500_df.head())
