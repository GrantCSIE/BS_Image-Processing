"""
Created on Thu Nov 16 14:17:41 2023

@author: CCU lab308 Shen Gwan-En
"""

def Laplacian_in_spatial(gray_img, L_max, L_min, count):
    """
    求原影像在Spatial domain上經Laplacian filter處理後的結果。
    
    Parameters
    ----------
    gray_img : array_like
            輸入影像對應的矩陣
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值
    count : int
            執行disp_img()函式時，顯示在圖表上的第幾列。
            
    Returns
    -------
    None.
    
    """
    
    transformed_img = elementwise_multi(gray_img, Laplacian, L_min, L_max - 1)
    
    sharpening_img =  img_addition(gray_img, transformed_img, 1, L_max, L_min)
    
    title = 'image after Laplacian filter \nin spatial domain'

    disp_img(sharpening_img, count, 1, title)
    
def Laplacian_in_frequency(img, L_max, L_min, count):
    """
    求原影像在frequency domain上經Laplacian filter處理後的結果。
    
    步驟:
        1. 求矩陣的Discrete Fourier transform，且原點已移至(M/2, N/2): DFT_img
        2. 求Laplacian filter: Filter
        3. 計算Filter * DFT_img
        4. 計算step 3的Inverse discrete Fourier transform: transformed_img
        5. 將經IDFT的結果正規化在intervel[-255,255]之間: normalized_img
        6. 將原影像和正規化的影像相加: result_img
        7. 顯示處理結果影像。
        
    Parameters
    ----------
    gray_img : array_like
            輸入影像對應的矩陣
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值
    count : int
            執行disp_img()函式時，顯示在圖表上的第幾列。
            
    Returns
    -------
    None.
    
    """
    
    DFT_img = np.fft.fft2(img)
    
    centered_DFT_img = freq_shifting(DFT_img)

    Filter = Laplacian_freq(centered_DFT_img)

    IDFT_img = freq_shifting(Filter * centered_DFT_img)
    
    transformed_img = np.real(np.fft.ifft2(IDFT_img))

    normalized_img = img_normalization(transformed_img, 255, -255)

    result_img = img_addition(gray_img, normalized_img, -1, L_max, L_min)
    
    title = 'image after Laplacian filter \nin frequency domain'
    
    disp_img(result_img, count, 2, title)


def rgb2gray(img):
    """
    求原影像對應的gray image。
    
    Parameters
    ----------
    img : array_like
            輸入影像對應的矩陣
            
    Returns
    -------
    (dim_x, dim_y) : tuple
        原影像的維度(x*y)，x和y皆為int
    
    gray_img :　array_like
        灰階影像對應的矩陣
    
    """
    
    (dim_x, dim_y) = img.shape[0], img.shape[1]
    gray_img = np.zeros((dim_x, dim_y), dtype = 'uint8') 
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            gray_img[i][j] = img[i][j][0]
    return (dim_x, dim_y), gray_img

def elementwise_multi(src_img, kernel, L_min, L_max):
    
    """
    求原影像中，取每一個subimage和mask進行元素間的相乘之結果。
    
    Parameters
    ----------
    src_img : array_like
            欲進行處理的影像對應的矩陣
            
    kernel : array_like
            mask對應的矩陣
    
    L_max : int
            灰階影像的gray level最大值
            
    L_min : int
            灰階影像的gray level最小值
            
    Returns
    -------
    
    transformed_img : array_like
            經過mask處理過後的影像
    
    """
    
    (dim_x, dim_y) = src_img.shape[0], src_img.shape[1]
    (kdim_x, kdim_y) = kernel.shape[0], kernel.shape[1]
    
    transformed_img = np.zeros((dim_x, dim_y))
    
    for i in range(dim_x - kdim_x + 1):
        
        for j in range(dim_y - kdim_y + 1):
            
            sub_img = gray_img[i : kdim_x + i, j : kdim_y + j]
            
            transformed_img[i][j] = np.sum(sub_img * kernel)
            
            if transformed_img[i][j] <= L_min:
                transformed_img[i][j] = L_min
                
            elif transformed_img[i][j] >= L_max:
                transformed_img[i][j] = L_max
                
    return transformed_img

def disp_img(img, col_index, row_index, title):
    """
    顯示原影像在圖表中。
    
    Parameters
    ----------
    img : array_like
            欲顯示的影像對應的矩陣。
            
    row_index : int
            影像所在圖表的列數。
    
    col_index : int
            影像所在圖表的行數。
            
    title: string
            顯示在圖表中影像上的文字。
            
    Returns
    -------
    
    None.
    
    """
    
    axes[row_index][col_index].imshow(img, cmap = 'gray')
    axes[row_index][col_index].set_title(title, fontsize = 15)
    axes[row_index][col_index].axis('off')


def freq_shifting(img):
    """
    將原影像shifting至center，或shifting至original。
        
    Parameters
    ----------
    img : array_like, only can be 2-dimensional
            欲移動的影像對應的矩陣。
            
    Returns
    -------
    img : array_like
        經移動的影像對應的矩陣。
        
    """
    
    (dim_x, dim_y) = img.shape[0], img.shape[1]
    
    for i in range(0, dim_x):
        img[i] = np.roll(img[i], dim_y // 2)
    
    for j in range(0, dim_y):
        img[:, j] = np.roll(img[:, j], dim_x // 2)  
            
    return img


def Laplacian_freq(freq_img):
    """
    求原影像size在frequency domiain上的Laplacian filter，且filter已移動至center。
        
    Parameters
    ----------
    freq_img : array_like, only can be 2-dimensional
            frequency domain上的影像對應的矩陣。
            
    Returns
    -------
    Filter : array_like
        已移動至center的Laplacian filter。
        
    """
    
    (dim_x, dim_y) = freq_img.shape[0], freq_img.shape[1]
    Filter = np.zeros((dim_x, dim_y), dtype = np.float32)
    
    for u in range(dim_x):
        for v in range(dim_y):
            Filter[u][v] = -4 * (np.pi * np.pi) * \
                (((u - (dim_x / 2)) ** 2)  + ((v - (dim_y / 2)) ** 2)) 
            
    return Filter

def img_normalization(img, upper, lower):
    """
    將原影像正規化在intervel[lower, upper]之間。
        
    Parameters
    ----------
    img : array_like, only can be 2-dimensional
            影像對應的矩陣。
            
    upper : int
        經正規化的影像的gray level上限。

    lower : int
        經正規化的影像的gray level下限。 
           
    Returns
    -------
    scaled_img : array_like
        經正規化的影像結果。
        
    """
    
    OldRange = np.max(img) - np.min(img)
    NewRange = upper - lower
    scaled_img = (((img - np.min(img)) * NewRange) / OldRange) + lower
    
    return scaled_img

def img_addition(img1, img2, const, L_max, L_min):
    """
    將兩張影像進行加法運算。
        
    Parameters
    ----------
    img1 : array_like, only can be 2-dimensional
            第一個影像對應的矩陣。
    img2 : array_like, only can be 2-dimensional
            第二個影像對應的矩陣。
    const : int
            第二張影像上的權重。
        
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值
           
    Returns
    -------
    addition_image : array_like
        經加法運算的影像對應的矩陣。
        
    """
    
    (dim_x, dim_y) = img1.shape[0], img1.shape[1]
    
    addition_image = np.zeros((dim_x, dim_y))
    
    for i in range(dim_x):
        for j in range(dim_y):
            
            addition_image[i][j] = img1[i][j] + const * img2[i][j] 
            
            if addition_image[i][j] >= L_max:
                addition_image[i][j] = L_max
                
            elif addition_image[i][j] <= L_min:
                addition_image[i][j] = L_min
    
    return addition_image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


count = 0

path = 'HW2_test_image/'

Laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype = 'int8')

(L_max, L_min) = 255, 0

fig, axes = plt.subplots(3,
                         len(glob.glob(path + '*')),
                         figsize = (25, 25)
                         ,squeeze=False)

if __name__ == '__main__':
    for files in glob.glob(path + '*'):
    
        img = cv2.imread(files)
        
        (dim_x, dim_y), gray_img = rgb2gray(img)
        
        title = 'original image'
        
        disp_img(gray_img, count, 0, title)
        
        Laplacian_in_spatial(gray_img, L_max, L_min, count)
        
        Laplacian_in_frequency(gray_img, L_max, L_min, count)
        
        count += 1
        
    plt.show()