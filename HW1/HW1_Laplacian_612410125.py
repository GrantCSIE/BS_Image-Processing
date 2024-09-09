# -*- coding: utf-8 -*-
"""
Spyder Editor

author: 612410125 Shen-Kwan En
"""
'''
function rgb2gray() 的功能為:
    轉換3維的gray image成2維的image.
    其中,傳入值:
        img:欲轉換的image.
    回傳值:
        無.
        
'''

def rgb2gray(img): 
    (dim_x, dim_y) = img.shape[0], img.shape[1] #dim_x 和 dim_y分別代表 image的長和寬
    if img.ndim == 3:
        gray_img = np.zeros((dim_x, dim_y), dtype = 'uint8') #建立一個dim_x* dim_y的zero matrix, datatype為length為8之無號int
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                gray_img[i][j] = img[i][j][0] #將img每個entries的第一個index的data取出,放在gray_img每個entries內
        return (dim_x, dim_y), gray_img
    else:
        return (dim_x, dim_y),img
    
'''
function disp_img() 的功能為:
    在圖表的第(x,y)位置列印出灰階image.
    其中,傳入值:
        x:紀錄image應繪製在圖表的哪一個row上,若為0則印在第一列, 為3則印在第四列.
        y:共有y = (HW1_test_image資料夾中image數) 張圖, 紀錄image應繪製在圖表的哪一個column上.
        img:欲列印的image.
        title:image上的標題,印出original image時標題為: "圖片名稱.副檔名"
                        ,印出經 Laplaction operator 之image時標題為: "圖片名稱.副檔名 after Laplaction operator"
                        ,印出經 unsharpening的image時標題為: "圖片名稱.副檔名 after unshapening"
    回傳值:
        無.
        
'''
def disp_img(x, y, img, title, L_min, L_max):
    axes[x][y].imshow(img, cmap = 'gray',vmin = L_min, vmax = L_max)
    axes[x][y].set_title(title, fontsize = font)
    axes[x][y].axis('off')
    
'''
function elementwise_multi()的功能為:
    計算一個m*n的matrix對一個3*3的kernel做卷積運算.
    其中,傳入值:
        src_img:欲進行卷積運算的matrix.
        kernel:卷積運算的kernel.
    回傳值:
        transformed_img:經卷積運算的matrix.
'''

def elementwise_multi(src_img, kernel, L_min, L_max):
    
    (dim_x, dim_y) = src_img.shape[0], src_img.shape[1]
    (kdim_x, kdim_y) = kernel.shape[0], kernel.shape[1]
    
    transformed_img = np.zeros((dim_x, dim_y), dtype = 'uint8')
    
    for i in range(dim_x - kdim_x + 1):
        for j in range(dim_y - kdim_y + 1):
            sub_img = gray_img[i : kdim_x + i, j : kdim_y + j]
            
            # 若convoluction結果<0, 則指定其為0,否則為原本之值
            if np.sum(sub_img * kernel) <= L_min:
                transformed_img[i][j] = L_min
            elif np.sum(sub_img * kernel) >= L_max:
                transformed_img[i][j] = L_max
            else:
                transformed_img[i][j] = np.sum(sub_img * kernel)
    return transformed_img

'''
function Sharpening()的功能為:
    將經Laplaction operator的image和原本灰階image相加.
    其中,傳入值:
        src_img:原本灰階image.
        transformed_img:經Laplaction operator的image.
    回傳值:
        sharpening_img:經相加後的image.
'''

def Sharpening(src_img, transformed_img, L_min, L_max):
    (dim_x, dim_y) = src_img.shape[0], src_img.shape[1]
    
    sharpening_img = np.zeros((dim_x, dim_y), dtype = 'uint16')
    
    for i in range(0, dim_x):
        for j in range(0, dim_y):       
                sharpening_img[i][j] = transformed_img[i][j]
                sharpening_img[i][j] += src_img[i][j]
        
                if sharpening_img[i][j] > L_max:
                    sharpening_img[i][j] = L_max
            
    return sharpening_img


import matplotlib.pyplot as plt
import numpy as np
import glob

path = 'HW1_test_image/'

Laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype = 'int8')

#建立subplot,size為(3*原始圖片數量)
fig, axes = plt.subplots(3,
                         len(glob.glob(path + '*.bmp')),
                         figsize = (25, 25))
font = 15 #圖表字型大小為15
count = 0 #從第1張圖開始計算
gray_level = pow(2, 8) #gray level從 0 至 2^8-1


if __name__ == '__main__':
    
    for files in glob.glob(path + '*.bmp'):
        
        img = plt.imread(files)
        
        (dim_x, dim_y), gray_img = rgb2gray(img)
        
        disp_img(0, count, gray_img,
                 f"{files[len('HW1_test_image/') : ]}",
                 0, gray_level - 1)
        
        transformed_img = elementwise_multi(gray_img, Laplacian, 0, gray_level - 1)
        
        disp_img(1, count, transformed_img,
                 f"{files[len('HW1_test_image/') : ]}\n after Laplaction operator",
                 0, gray_level - 1)
        
        sharpening_img = Sharpening(gray_img, transformed_img, 0, gray_level - 1)
        
        disp_img(2, count, sharpening_img,
                 f"{files[len('HW1_test_image/') : ]}\n after shapening",
                 0, gray_level - 1)
        
        count += 1 #下一張圖
    plt.show()