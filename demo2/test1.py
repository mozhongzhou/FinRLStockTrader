import pandas as pd

# 读取CSV文件
df = pd.read_csv("e:/Coding/FinRLStockTrader/demo2/results/a2c_actions.csv")

# 找出至少有一个非零值的行
non_zero_rows = df[df.ne(0).any(axis=1)]

# 显示这些行
print(non_zero_rows)

# 获取这些行的索引
row_indices = non_zero_rows.index.tolist()
print("非零行的索引:", row_indices)
