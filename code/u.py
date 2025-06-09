import numpy as np
from math import ceil

# 定义常量
M = 2
P = np.array([2016,2048,2304,2520],int)  # 使用 numpy 数组
mul = np.array([12728,16836,15868,16540],int) 
# 计算 L 和 S
L = P - M + 1
S = np.array(np.ceil(1500 / L),int ) # 使用 numpy 的 ceil 函数对数组进行操作
T = S*(mul*2+P*3)
# 堆叠结果并打印
result = np.stack([P, L, S,T], axis=1)  # 堆叠 P, L, S 为二维数组
print(result)