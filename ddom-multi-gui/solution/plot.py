import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 文件名列表
files = ['output_314159.csv', 'output_1469983670.csv', 'output_1569983670.csv']

# 列名列表
columns = ['value_max_1', 'value_max_2', 'value_max_3', 'value_mean_1', 'value_mean_2', 'value_mean_3']

# 初始化DataFrame用于存储平均值和标准差
df_mean = pd.DataFrame(columns=columns)
df_std = pd.DataFrame(columns=columns)

# 读取文件并计算平均值和标准差
for col in columns:
    data = [pd.read_csv(file)[col] for file in files]
    data_concat = pd.concat(data, axis=1)
    df_mean[col] = data_concat.mean(axis=1)
    df_std[col] = data_concat.std(axis=1)

# 保存平均值和标准差到CSV文件
df_mean.to_csv('mean_values.csv', index=False)
df_std.to_csv('std_values.csv', index=False)

# 准备绘图
step = np.linspace(0, 1000, len(df_mean))

# 绘制图形
plt.figure(figsize=(9, 6))

labels = [
    'log P max value', 
    'QED max value', 
    'SA min value', 
    'log P mean value', 
    'QED mean value', 
    'SA mean value'
]

colors = ['lightblue', 'dodgerblue', 'navy', 'salmon', 'firebrick', 'darkred']

for col, label, color in zip(columns, labels, colors):
    plt.plot(step, df_mean[col], label=f'{label}', color=color)
    plt.fill_between(step, df_mean[col] - df_std[col], df_mean[col] + df_std[col], color=color, alpha=0.2)

# 添加标题和标签
# plt.title('Diffusion process')
plt.xlabel('Time step', fontsize=22)
plt.ylabel('Obj value', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((-0.02,1.02))

# 显示图例
# plt.legend(fontsize=18, loc='center right', bbox_to_anchor=(1.2, 0.3))
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)
plt.subplots_adjust(top=0.8)  # 调整数字以适应你的需求


# 显示图形
plt.savefig("guidance.jpg")
plt.savefig("guidance.pdf")
