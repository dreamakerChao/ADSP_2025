import sys
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_cie76

# Bayer matrix generation

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller_matrix = generate_bayer_matrix(n - 1)
        size = 2 ** n
        new_matrix = np.zeros((size, size), dtype=int)
        for i in range(2 ** (n - 1)):
            for j in range(2 ** (n - 1)):
                base_value = 4 * smaller_matrix[i, j]
                new_matrix[i, j] = base_value
                new_matrix[i, j + 2 ** (n - 1)] = base_value + 2
                new_matrix[i + 2 ** (n - 1), j] = base_value + 3
                new_matrix[i + 2 ** (n - 1), j + 2 ** (n - 1)] = base_value + 1
        return new_matrix

def generate_thresholds_matrix(bayer_matrix):
    N = bayer_matrix.shape[0]
    thresholds_matrix = np.zeros_like(bayer_matrix, int)
    for i in range(N):
        for j in range(N):
            thresholds_matrix[i, j] = (255 * (bayer_matrix[i, j] + 0.5)) / (N ** 2)
    return thresholds_matrix

# Ordered Dithering

def ordered_dithering(img, thresholds_matrix):
    N = thresholds_matrix.shape[0]
    output_img = np.zeros_like(img, np.uint8)
    height, width = img.shape
    for i in range(0, height, N):
        for j in range(0, width, N):
            for k in range(N):
                for l in range(N):
                    if i + k < height and j + l < width:
                        if img[i + k, j + l] > thresholds_matrix[k, l]:
                            output_img[i + k, j + l] = 255
                        else:
                            output_img[i + k, j + l] = 0
    return output_img

# Error Diffusion with different kernels and optional adaptive threshold

def error_diffusion(img, kernel_name='floyd', adaptive=False):
    kernels = {
        'floyd': np.array([[0, 0, 7], [3, 5, 1]]) / 16.0,
        'jarvis': np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]) / 48.0,
        'stucki': np.array([[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]]) / 42.0,
        'burkes': np.array([[0, 0, 0, 8, 4], [2, 4, 8, 4, 2]]) / 32.0
    }
    kernel = kernels.get(kernel_name, kernels['floyd'])
    kH, kW = kernel.shape
    offset = kW // 2

    if adaptive:
        thresh = cv.adaptiveThreshold(img.astype(np.uint8), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 11, 2)
    else:
        thresh = np.full(img.shape, 128)

    img = img.astype(float)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > thresh[i, j] else 0
            img[i, j] = new_pixel
            error = old_pixel - new_pixel
            for ki in range(kH):
                for kj in range(kW):
                    ni, nj = i + ki, j + kj - offset
                    if 0 <= ni < height and 0 <= nj < width:
                        img[ni, nj] += error * kernel[ki, kj]
    return np.clip(img, 0, 255).astype(np.uint8)

# PSNR

def calculate_psnr(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

# SSIM using grayscale

def calculate_ssim(original, compressed):
    gray_orig = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    gray_comp = cv.cvtColor(compressed, cv.COLOR_BGR2GRAY)
    return ssim(gray_orig, gray_comp, data_range=255)

# Delta E76

def calculate_deltaE2000(original, compressed):
    lab1 = rgb2lab(original)
    lab2 = rgb2lab(compressed)
    deltaE = deltaE_cie76(lab1, lab2)
    return np.mean(deltaE)

# Main execution

if __name__ == '__main__':
    img_rgb = cv.imread("uk2.jpg")
    thresholds_matrix = generate_thresholds_matrix(generate_bayer_matrix(2))

    method_configs = [
        ('ordered_dithering', {}),
        ('error_diffusion', {'kernel_name': 'floyd'}),
        ('error_diffusion', {'kernel_name': 'jarvis'}),
        ('error_diffusion', {'kernel_name': 'stucki'}),
        ('error_diffusion', {'kernel_name': 'burkes'}),
        ('error_diffusion', {'kernel_name': 'floyd', 'adaptive': True}),
    ]

    for method, kwargs in method_configs:
        channels_halftoned = []
        for i in range(3):
            if method == 'ordered_dithering':
                channel_halftoned = ordered_dithering(img_rgb[..., i], thresholds_matrix)
            elif method == 'error_diffusion':
                channel_halftoned = error_diffusion(img_rgb[..., i], **kwargs)
            channels_halftoned.append(channel_halftoned)

        halftoned_rgb = cv.merge(channels_halftoned)
        out_name = method + '_' + '_'.join(f"{k}{v}" for k, v in kwargs.items())
        cv.imwrite(f'{out_name}.png', halftoned_rgb)

        psnr = calculate_psnr(img_rgb, halftoned_rgb)
        ssim_val = calculate_ssim(img_rgb, halftoned_rgb)
        deltaE = calculate_deltaE2000(img_rgb, halftoned_rgb)

        print(f"{out_name.upper()} Results:")
        print(f"  PSNR: {psnr:.2f}")
        print(f"  SSIM: {ssim_val:.4f}")
        print(f"  ΔE: {deltaE:.2f}\n")
