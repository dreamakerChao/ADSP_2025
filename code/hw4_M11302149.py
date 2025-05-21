import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

def rgb_to_ycbcr(image):
    # Manual RGB to YCbCr conversion (BT.601 standard)
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)

    # YCbCr conversion formula
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    # Manual YCbCr to RGB conversion (BT.601 inverse)
    Y = Y.astype(np.float32)
    Cb = Cb - 128
    Cr = Cr - 128

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    # Clip values to [0, 255] and stack
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    return np.stack((R, G, B), axis=-1).astype(np.uint8)

def downsample_420(channel):
    # Downsample both horizontally and vertically by factor of 2
    return channel[::2, ::2]

def upsample_bilinear(channel, target_shape):
    # Upsample using OpenCV's resize with bilinear interpolation
    return cv2.resize(channel, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def C420(A):
    # Step 1: Convert RGB to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(A)

    # Step 2: 4:2:0 downsampling on Cb and Cr
    Cb_down = downsample_420(Cb)
    Cr_down = downsample_420(Cr)

    # Step 3: Bilinear upsampling to original size
    Cb_up = upsample_bilinear(Cb_down, Y.shape)
    Cr_up = upsample_bilinear(Cr_down, Y.shape)

    # Step 4: Convert back to RGB
    B = ycbcr_to_rgb(Y, Cb_up, Cr_up)

    # Step 5: Calculate PSNR
    psnr_value = psnr(A, B)
    print(f"PSNR between original and reconstructed image: {psnr_value:.2f} dB")

    return B

if __name__ == "__main__":
    # Example usage
    # Load an image (make sure to replace 'image.jpg' with your image path)
    A = cv2.imread('image.jpg')
    if A is None:
        print("Error: Image not found.")
    else:
        # Convert BGR to RGB
        A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

        # Process the image
        print("Processing image...")
        B = C420(A)

        # Save the result
        cv2.imwrite('output_image.jpg', cv2.cvtColor(B, cv2.COLOR_RGB2BGR))
