# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:17:41 2023

@author: CCU lab308 Shen Gwan-En
"""

def high_boost_filtering_spatial(img, kdim_x, kdim_y, count, A, L_max, L_min):
    """
    求原影像在Spatial domain上經high boost filtering後的結果。
    
    Parameters
    ----------
    img : array_like
            輸入影像對應的矩陣
    kdim_x: int
            和變數kdim_y同時決定averaging filter的維度。
    kdim_y: int
            和變數kdim_x同時決定averaging filter的維度。
    count : int
            執行disp_img()函式時，顯示在圖表上的第幾列。
    A : int
            常數A，決定filtering結果中，原影像的占比。通常大於1。
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值
            
    Returns
    -------
    None.
    
    """
    blur_img = blurring(img, kdim_x, kdim_y)
    
    unsharp_mask = img_subtraction(img, blur_img, L_max, L_min)
    
    high_boost_img = img_addition(unsharp_mask, gray_img, (A - 1), L_max, L_min)

    title = 'image after high boost filtering \nin spatial domain'
    
    disp_img(high_boost_img, 1, count, title)

def high_boost_filtering_frequency(img, radius, count, A, L_max, L_min):
    """
    求原影像在frequency domain上經high boost filtering後的結果。
    
    步驟:
        1. 求矩陣的Discrete Fourier transform: DFT_img
        2. 將經DFT的影像的原點移至(M/2, N/2) : centered_DFT_img
        3. 求得半徑為radius的Ideal lowpass filter : Filter
        4. 將Filter和原影像相乘後的結果移回(0,0) : IDFT_img
        5. 求step 4的結果的inverse discrete Fourier transform，
            並取實數部分，得到原影像經Lowpass filter的結果 : low_pass_img
        6. 將原影像減去原影像經Lowpass filter的結果 : unsharp_mask
        7. 將原影像乘上常數A後，和step 6的結果相加 : high_boost_img
        8. 顯示處理結果影像。
        
    Parameters
    ----------
    img : array_like
            輸入影像對應的矩陣
    radius : int
            Lowpass filter的半徑D0。
    count : int
            執行disp_img()函式時，顯示在圖表上的第幾列。
    A : int
            常數A，決定filtering結果中，原影像的占比。通常大於1。
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值
            
    Returns
    -------
    None.
    
    """
    DFT_img = np.fft.fft2(img)
    
    centered_DFT_img = freq_shifting(DFT_img)
    
    Filter = ILPF(centered_DFT_img, radius)
    
    IDFT_img = freq_shifting(Filter * centered_DFT_img)
    
    low_pass_img = np.real(np.fft.ifft2(IDFT_img))

    unsharp_mask = img_subtraction(gray_img, low_pass_img, L_max, L_min)
    
    high_boost_img = img_addition(unsharp_mask, gray_img, (A - 1), L_max, L_min)
    
    title = 'image after high boost filtering \nin frequency domain'
    
    disp_img(high_boost_img, 2, count, title)
    

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
    if img.ndim == 3:
        gray_img = np.zeros((dim_x, dim_y), dtype = 'uint8') 
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                gray_img[i][j] = img[i][j][0]
        return (dim_x, dim_y), gray_img
    else:
        return (dim_x, dim_y),img
       

def disp_img(img, row_index, col_index, title):
    """
    顯示原影像在圖表中。
    
    Parameters
    ----------
    img : array-like
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

def padding(img):
    """
    將(x*y)大小的原影像向外圍擴增一圈形成(x+2*y+2)大小的新影像，
    將原影像放在新影像的中間區域，
    外圍區域每個pixel的gray level由以下方法決定：
    1.若其位於第一列，則將新影像的第二列(原影像的第一列)複製到新影像的第一列。
    2.若其位於最後一列，則將新影像的倒數第二列(原影像的最後一列)複製到新影像的最後一列。
    3.若其位於第一行，則將新影像的第二行(原影像的第一行)複製到新影像的第一行。
    4.若其位於最後一行，則將新影像的倒數第二行(原影像的最後一行)複製到新影像的最後一行。
    5.若其位於左上角，則將新影像的[第二列,第二行]對應的gray level(原影像的左上角)複製到新影像的左上角。
    6.若其位於右上角，則將新影像的[第二列,倒數第二行]對應的gray level(原影像的右上角)複製到新影像的右上角。
    7.若其位於左下角，則將新影像的[倒數第二列,第二行]對應的gray level(原影像的左下角)複製到新影像的左下角。
    8.若其位於右下角，則將新影像的[倒數第二列,倒數第二行]對應的gray level(原影像的右下角)複製到新影像的右下角。
    
    Parameters
    ----------
    img : array-like
            欲顯示的影像對應的矩陣。

    Returns
    -------
    (padding_img.shape[0], padding_img.shape[1]) : tuple
            經擴增的影像的維度。
    padding_img : array-like
        經擴增的影像。

    """
    (dim_x, dim_y) = img.shape[0], img.shape[1] 
    padding_img = np.zeros((dim_x + 2, dim_y + 2))
    
    #將原影像放在新影像中間部分,外圍一圈留空
    for i in range(1, dim_x + 1):
        for j in range(1, dim_y + 1):
            padding_img[i][j] = img[i - 1][j - 1]
            
    #對最上排第2項到倒數第2項做padding，每次取對應下方的pixel的intensity
    for j in range(1, dim_y + 1):
        padding_img[0][j] = padding_img[1][j]
        
    #對最左行第2項到倒數第2項做padding，每次取對應右方的pixel的intensity
    for i in range(1, dim_x + 1):
        padding_img[i][0] = padding_img[i][1]
        
    #對最右行第2項到倒數第2項做padding，每次取對應左方的pixel的intensity 
    for i in range(1, dim_x + 1):
        padding_img[i][dim_y + 1] = padding_img[i][dim_y]
    
    #對最下排第2項到倒數第2項做padding，每次取對應上方的pixel的intensity
    for j in range(1, dim_y + 1):
        padding_img[dim_x + 1][j] = padding_img[dim_x][j]
        
    #左上角的intensity由其右下角取代   
    padding_img[0][0] = padding_img[1][1]   
    
    #右上角的intensity由其左下角取代
    padding_img[0][dim_y + 1] = padding_img[1][dim_y]    
    
    #左下角的intensity由其右上角取代
    padding_img[dim_x + 1][0] = padding_img[dim_x][1]        
    
    #右下角的intensity由其左上角取代
    padding_img[dim_x + 1][dim_y + 1] = padding_img[dim_x][dim_y]
    
    return (padding_img.shape[0], padding_img.shape[1]), padding_img

def blurring(img, kdim_x, kdim_y):
    """
    求原影像對應的模糊化影像。

    Parameters
    ----------
    img : array-like
        欲求模糊化影像對應原影像。
    kdim_x: int
            和變數kdim_y同時決定averaging filter的維度。
    kdim_y: int
            和變數kdim_x同時決定averaging filter的維度。
    Returns
    -------
    blurring_img : array-like
        所求之模糊化影像。

    """
    
    
    (dim_x, dim_y) = img.shape[0], img.shape[1]
    
    padding_img = img
    for i in range(0, max(kdim_x, kdim_y)):
        (dim_x, dim_y), padding_img = padding(padding_img)
    
    kernel = np.ones((kdim_x, kdim_y)) / (kdim_x * kdim_y)
    
    blurring_img = elementwise_multi(img, padding_img, kernel, 0, 255)
    
    return blurring_img

def elementwise_multi(src_img, padding_img, kernel, L_min, L_max):
    """
    求原影像中，取每一個subimage和mask進行元素間的相乘之結果。
    此外，會將結果影像的大小修改成和原影像相同大小。
    
    Parameters
    ----------
    src_img : array-like
            欲進行處理的影像對應的矩陣。
    padding_img: array-like
            經擴增大小的影像對應的矩陣。
    kernel : array-like
            mask對應的矩陣。
    L_min : int
            灰階影像的gray level最小值
    L_max : int
            灰階影像的gray level最大值
            
            
    Returns
    -------
    
    transformed_img : array_like
            經過mask處理過後的影像
    
    """
    (dim_x, dim_y) = padding_img.shape[0], padding_img.shape[1]
    (kdim_x, kdim_y) = kernel.shape[0], kernel.shape[1]
    
    tmp_img = np.zeros((dim_x - kdim_x + 1, dim_y - kdim_y + 1))
    
    for i in range(dim_x - kdim_x + 1):
        for j in range(dim_y - kdim_y + 1):
            
            sub_img = padding_img[i : kdim_x + i, j : kdim_y + j]
            
            if np.sum(sub_img * kernel) <= L_min:
                tmp_img[i][j] = L_min
            elif np.sum(sub_img * kernel) >= L_max:
                tmp_img[i][j] = L_max
            else:
                tmp_img[i][j] = np.sum(sub_img * kernel)
                
    transformed_img = np.zeros((src_img.shape[0], src_img.shape[1]))
    
    for i in range(0, src_img.shape[0]):
        for j in range(0, src_img.shape[1]):
            transformed_img[i][j] = tmp_img[i + int(kdim_x/2)][j + int(kdim_y/2)]
            
    return transformed_img
    

def img_subtraction(img1, img2, L_max, L_min):
    """
    將兩張影像進行減法運算。

    Parameters
    ----------
    img1 : array-like
        欲進行減法運算的矩陣。
    img2 : array-like
        欲進行減法運算的矩陣。
    L_max : int
            灰階影像的gray level最大值
    L_min : int
            灰階影像的gray level最小值

    Returns
    -------
    subtracted_img : array-like
        減法運算的結果影像。

    """
    
    (dim_x, dim_y) = img1.shape[0], img2.shape[1]
    
    subtracted_img = np.zeros((dim_x, dim_y))
    
    for i in range(dim_x):
        for j in range(dim_y):
            
            subtracted_img[i][j] = img1[i][j] - img2[i][j]
            
            if subtracted_img[i][j] >= L_max:
                subtracted_img[i][j] = L_max
                
            elif subtracted_img[i][j] <= L_min:
                subtracted_img[i][j] = L_min
                
    return subtracted_img

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
    
  
    for i in range(0, dim_x):#對每一列做shifiting
        img[i] = np.roll(img[i], dim_y // 2)
    
    for j in range(0, dim_y):#對每一行做shifiting
        img[:, j] = np.roll(img[:, j], dim_x // 2)   
        
    return img


def ILPF(img, radius):
    """
        求得半徑D0為radius的Ideal lowpass filter。

    Parameters
    ----------
    img : array-like
        原影像對應的矩陣
    radius : int
        Ideal lowpass filter的半徑長。

    Returns
    -------
    Filter : array-like
        對應的Ideal lowpass filter。

    """
    
    (dim_x, dim_y) = img.shape[0], img.shape[1]
    
    D = np.zeros((dim_x, dim_y), dtype = np.float32)
    Filter = np.zeros((dim_x, dim_y), dtype = np.float32)
    
    for u in range(dim_x):
        for v in range(dim_y):
            D[u][v] = pow((u - dim_x / 2) ** 2 + (v - dim_y / 2) ** 2, 0.5)
            if D[u][v] <= radius:
                Filter[u][v] = 1
            else:
                Filter[u][v] = 0
    return Filter

import matplotlib.pyplot as plt
import numpy as np
import glob

count = 0
radius = 50
A = 1.7
L_max = 255
L_min = 0

path = 'HW2_test_image/'

fig, axes = plt.subplots(3,
                         len(glob.glob(path + '*')),
                         figsize = (25, 25))

if __name__ == '__main__':
    for files in glob.glob(path + '*'):
        
        img = plt.imread(files)
        
        (dim_x, dim_y), gray_img = rgb2gray(img)
        
        title = 'original image'
        
        disp_img(gray_img, 0, count, title)
        
        high_boost_filtering_spatial(gray_img, 3, 3, count, A, L_max, L_min)
        
        high_boost_filtering_frequency(gray_img, radius,count, A, L_max, L_min)
        
        count += 1
    plt.show()