import sys
import cv2 as cv
import numpy as np
import matplotlib as plt

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
            thresholds_matrix[i,j] = (255*(bayer_matrix[i,j]+0.5))/(N**2)

    return thresholds_matrix


def Ordered_Dithering(img, thresholds_matrix):
    N = thresholds_matrix.shape[0]
    Ordered_Dithering_img = np.zeros_like(img,np.uint8)
    height,width = img.shape

    for i in range(0,img.shape[0],N):
        for j in range(0,img.shape[1],N):
            for k in range(N):
                for l in range(N):
                    if (i + k < height and j + l < width):
                        if(img[i+k,j+l]>thresholds_matrix[k,l]):
                            Ordered_Dithering_img[i+k,j+l] = 255
                        else:
                            Ordered_Dithering_img[i+k,j+l] = 0

    return Ordered_Dithering_img

def Error_Diffusion(img,kernel=2):
    Error_Diffusion_img = img.astype(float).copy()
    height, width = img.shape
    
    
    if(kernel==2):
        '''
        kernel:
        1/16 
        [   0  x  7
            3  5  1    ]
        '''
        for i in range(height - 1):
            for j in range(width - 1):
                old_pixel = Error_Diffusion_img[i, j]

                if( old_pixel > 128):
                    new_pixel = 255
                else:
                    new_pixel = 0 

                Error_Diffusion_img[i, j] = new_pixel
                error = old_pixel - new_pixel

                # diffusion
                if(j+1<width):
                    Error_Diffusion_img[i, j+1] += (error * 7) /16
                if(i+1<height and j-1>=0):
                    Error_Diffusion_img[i + 1, j - 1] += (error * 3) /16
                if(i+1<height):
                    Error_Diffusion_img[i+1, j] += (error * 5 )/16
                if(i+1<height and j+1<width ):
                    Error_Diffusion_img[i+1, j+1] += (error * 1) /16
    else:

        kernel = np.array([
            [0,  0,  0,  7,  5],
            [3,  5,  7,  5,  3],
            [1,  3,  5,  3,  1]
            ]) / 48.0


        for i in range(height - 2): 
            for j in range(2, width - 2):
                old_pixel = Error_Diffusion_img[i, j]
                new_pixel = 255 if old_pixel > 128 else 0

                Error_Diffusion_img[i, j] = new_pixel
                error = old_pixel - new_pixel

                for ki in range(3):
                    for kj in range(-2, 3):
                        ni, nj = i + ki, j + kj
                        if 0 <= ni < height and 0 <= nj < width:
                            Error_Diffusion_img[ni, nj] += error * kernel[ki, kj + 2]

    return np.clip(Error_Diffusion_img, 0, 255).astype(np.uint8)


def calculate_PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return np.infty
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


if __name__ == '__main__':

    #img = cv.imread(sys.argv[1])
    img = cv.imread("F-16-image.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("Baboon-image.png", cv.IMREAD_GRAYSCALE)
    #cv.imshow('Grayscale Image',img2 )
    #cv.waitKey(0) 
    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)
    output01 = Ordered_Dithering(img,thresholds_matrix)
    output02= Ordered_Dithering(img2,thresholds_matrix)
    output11 = Error_Diffusion(img)
    output12= Error_Diffusion(img2)
    output13 = Error_Diffusion(img,5)
    output14= Error_Diffusion(img2,5)

    #cv.imshow('Grayscale Image',output )


    #cv.waitKey(0)  
    #cv.destroyAllWindows()  # 關閉所有視窗

    cv.imwrite('F-16-image_order_dithering.png', output01)
    print(f"PSNR_o1: {calculate_PSNR(img,output01)}\n")
    cv.imwrite('Baboon-image_order_dithering.png', output02)
    print(f"PSNR_o2: {calculate_PSNR(img2,output02)}\n")

    cv.imwrite('F-16-image_error_diffusion.png', output11)
    print(f"PSNR_e1: {calculate_PSNR(img,output11)}\n")
    cv.imwrite('Baboon-image_diffusion.png', output12)
    print(f"PSNR_e2: {calculate_PSNR(img2,output12)}\n")

    cv.imwrite('F-16-image_error_diffusion_5.png', output13)
    print(f"PSNR_e1: {calculate_PSNR(img,output13)}\n")
    cv.imwrite('Baboon-image_diffusion_5.png', output14)
    print(f"PSNR_e2: {calculate_PSNR(img2,output14)}\n")