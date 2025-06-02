import numpy as np

def fftereal(x, y):
    """
    計算兩個實值訊號 x 和 y 的 N 點 FFT,只呼叫一次 N 點 FFT。
    參數：
        x: 長度為 N 的一維 numpy 陣列，實數
        y: 長度為 N 的一維 numpy 陣列，實數
    回傳：
        Fx: x 的 N 點 FFT,長度為 N 的複數陣列
        Fy: y 的 N 點 FFT,長度為 N 的複數陣列
    """
    # 確認 x 和 y 的形狀一致
    if x.shape != y.shape:
        raise ValueError("Shape of x and y must be the same.")

    # 將 x, y 合成複數序列 z：z[n] = x[n] + j*y[n]
    z = x.astype(np.complex128) + 1j * y.astype(np.complex128)

    # 對複數序列 z 進行一次 N 點 FFT
    Z = np.fft.fft(z)

    # 計算 Z 的複共軛值
    Z_conj = np.conj(Z)

    # 建立索引陣列，用於對稱頻點 (−k mod N)
    N = Z.shape[0]
    k = np.arange(N)
    k_conj = (-k) % N

    # 取得「對稱頻點的複共軛」：Z_conj_flip[k] = conj(Z[(−k) mod N])
    Z_conj_flip = Z_conj[k_conj]

    # 依照實值訊號 FFT 還原公式還原 Fx, Fy：
    #   Fx[k] = (Z[k] + conj(Z[−k])) / 2
    #   Fy[k] = (Z[k] − conj(Z[−k])) / (2j)
    Fx = 0.5 * (Z + Z_conj_flip)
    Fy = (0.5 / 1j) * (Z - Z_conj_flip)

    return Fx, Fy


# 範例使用
if __name__ == "__main__":
    print("x is a 1 Hz sine wave, y is a 2 Hz cosine wave.\nonly one FFT is called.")
    N = 8
    n = np.arange(N)
    x = np.sin(2 * np.pi * 1 * n / N)  # 1 Hz 的正弦波
    y = np.cos(2 * np.pi * 2 * n / N)  # 2 Hz 的餘弦波

    Fx, Fy = fftereal(x, y)

    # 印出結果
    print("Fx (x 的 FFT):")
    print(Fx)
    print("\nFy (y 的 FFT):")
    print(Fy)
