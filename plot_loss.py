import re
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 定义存储 loss 值的列表
loss_values = []


# 打开 loss.txt 文件，以只读模式
with open('loss.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 去除行尾的换行符
        line = line.strip()
        # 检查行是否包含 'loss': 字段
        if "'loss': " in line:
            # 从字符串中提取 loss 值
            start_index = line.find("'loss': ") + len("'loss': ")
            end_index = line.find(',', start_index)
            if end_index == -1:
                end_index = line.find('}', start_index)
            try:
                # 提取 loss 值并转换为浮点数
                loss = float(line[start_index:end_index])
                # 将 loss 值添加到列表中
                loss_values.append(loss)
            except ValueError:
                continue

# 打印 loss 值列表
print(len(loss_values))

# 绘制折线图，设置折线颜色为红色，线条宽度为 1
plt.plot(loss_values, color='green', linewidth=1)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('SFT Loss Trend')
plt.grid(True)

# 设置图片长宽比为 1:2
fig = plt.gcf()
fig.set_size_inches(12, 6)

# 保存图片到指定路径
path = "deepseek_sft.png"
plt.savefig(path)