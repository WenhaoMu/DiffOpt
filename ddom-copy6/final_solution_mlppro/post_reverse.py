import pandas as pd
import numpy as np

# 读取CSV文件并跳过第一行
# task = "superconductor"
# task = "ant"
# task = "dkitty"
# task = "tf-bind-8"
# task = "tf-bind-10"
task = "chembl"
# df = pd.read_csv("superconductor_fourier.csv", skiprows=0)
df = pd.read_csv(f"{task}_mlppro.csv")

# 创建一个空的DataFrame用于存储序号、平均值和标准差
result_df = pd.DataFrame(columns=["Index", "Average", "Standard Deviation"])

# 每组有17行数据，共5组
rows_per_group = 17
num_groups = 5

df = df.tail(rows_per_group * num_groups)
print(df)

# 遍历每一行来计算所有组的平均值和标准差
for i in range(rows_per_group):
    # 提取所有组中第i行的第一列（假设第一列名为"Index"）的序号
    index = df.iloc[i, 0]
    
    # 提取所有组中第i行的第五列（假设第五列名为"Useful_Data"）的有用数据
    useful_data = df.iloc[i::rows_per_group, 4]

    # 计算平均值和标准差
    avg = np.mean(useful_data)
    std = np.std(useful_data)
    if i == 0:
        print(useful_data)

    # 将结果添加到result_df
    result_df = result_df.append({"Index": index, "Average": avg, "Standard Deviation": std}, ignore_index=True)

# 将结果保存到新的CSV文件
# result_df.to_csv("zr_superconductor_512.csv", index=False)
result_df.to_csv(f"zr_{task}_512.csv", index=False)

