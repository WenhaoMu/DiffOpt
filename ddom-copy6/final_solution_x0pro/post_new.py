import pandas as pd
import numpy as np

# 读取CSV文件
# task = "superconductor"
# task = "ant"
# task = "dkitty"
# task = "tf-bind-8"
# task = "tf-bind-10"
task = "chembl"
df = pd.read_csv(f"{task}_x0.csv")

# 创建一个空的DataFrame用于存储序号、平均值和标准差
result_df = pd.DataFrame(columns=["Index", "Average", "Standard Deviation"])

# 每组有17行数据
rows_per_group = 17

# 定义一个包含所需组数的列表
groups_to_use = [1, 2, 3, 5, 7]  # 第1, 2, 3, 5, 7组

# 遍历每一行来计算指定组的平均值和标准差
for i in range(rows_per_group):
    # 提取所有组中第i行的第一列（假设第一列名为"Index"）的序号
    index = df.iloc[i, 0]

    # 创建一个空列表来收集每个组的有用数据
    useful_data_list = []

    # 遍历所选的组
    for group in groups_to_use:
        # 计算该组在DataFrame中的行索引
        row_idx = (group - 1) * rows_per_group + i
        useful_data_list.append(df.iloc[row_idx, 4])

    # 转换为NumPy数组并计算平均值和标准差
    useful_data_array = np.array(useful_data_list)
    avg = np.mean(useful_data_array)
    std = np.std(useful_data_array)

    # 将结果添加到result_df
    result_df = result_df.append({"Index": index, "Average": avg, "Standard Deviation": std}, ignore_index=True)

# 将结果保存到新的CSV文件
result_df.to_csv(f"zn_{task}_512.csv", index=False)
