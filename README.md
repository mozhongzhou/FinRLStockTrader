目录名称 功能描述
data/raw_data/ 存储原始数据
data/processed_data/ 存储清洗后的增加了技术指标的数据
trained_models/ 存储训练好的模型文件（如.zip 格式）
results/ 存储回测结果，如收益曲线、风险指标等

process_data.ipynb 获取标普五百成分股 2000-01-01 到 2025-03-13 的数据
train_model.ipynb 搭建环境 训练模型
back_test.ipynb 回测 可视化结果

README.md 项目说明文档，包含安装和运行指导
.gitignore 忽略不必要的文件，如临时文件或大体积数据

使用 FinRL 框架
