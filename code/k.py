# Python (使用函式庫生成 W8 並構造 W16)
import numpy as np
from scipy.linalg import hadamard

# 使用 scipy 的 Hadamard 函數生成 W8
W8 = hadamard(8)

# 根據提示構造 W16
W16 = np.block([
    [W8, W8],
    [W8, -W8]
])

print("16-point Walsh transform matrix:")
print(W16)