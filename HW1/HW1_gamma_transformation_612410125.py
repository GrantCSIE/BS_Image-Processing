# -*- coding: utf-8 -*-
"""
Spyder Editor

author: 612410125 Shen-Kwan En
"""
'''
function gamma_transformation()的功能為:
    將matrix中每個entries取r次方.
    其中,傳入值:
        img:欲進行gamma transformation的matrix.
        gamma:對每個entries取r = gamma次方.
    回傳值:
        gamma_img:經gamma transformation的matirx.
'''

def gamma_transformation(img, gamma):
    gamma_img = np.zeros((dim_x, dim_y), dtype = 'float32')
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            #對每個pixel上的intensity取gamma次方
            gamma_img[i][j] = pow(1/ 255 * img[i][j], gamma) * 255
    gamma_img = np.array(gamma_img, dtype = 'uint8')
    return gamma_img

'''
function disp_img() 的功能為:
    在圖表的第(x,y)位置列印出灰階image.
    其中,傳入值:
        x:紀錄image應繪製在圖表的哪一個row上,若為0則印在第一列, 為3則印在第四列.
        y:共有y = (HW1_test_image資料夾中image數) 張圖, 紀錄image應繪製在圖表的哪一個column上.
        img:欲列印的image.
        title:image上的標題,印出original image時標題為: "original image"
                        ,印出經 gamma為r的gamma transformation結果時 之image時標題為: "transformed image (gamma = r)"
    回傳值:
        無.
'''

def disp_img(x, y, img, title): #顯示img在sunplot內,位置在(x,y)上,標題名稱為title
    axes[x][y].imshow(img, cmap = 'gray', vmin= 0, vmax= 255)
    axes[x][y].set_title(title)
    axes[x][y].axis('off')


'''
function rgb2gray() 的功能為:
    轉換3維的gray image成2維的image.
    其中,傳入值:
        img:欲轉換的image.
    回傳值:
        無.
        
'''

def rgb2gray(img): #將img轉換成2維matrix
    if img.ndim == 3: #若原image是3維,則要降成2維matrix
        (dim_x, dim_y) = img.shape[0], img.shape[1] 
        #建立一個(dim_x* dim_y)的zero matrix
        gray_img = np.zeros((dim_x, dim_y), dtype = 'uint16') 
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                #由於3維的image中三個entries其值相同，
                #故將img共(dim_x*dim_y)個pixels的三個元素中第一個元素取出,放在gray_img每個對應的pixel上
                gray_img[i][j] = img[i][j][0] 
        return gray_img
    
    else:#若原image是2維,不須做任何變動
        return img

import numpy as np
import matplotlib.pyplot as plt
import glob

count = 0 #此處為計算輸入image數量

gamma = np.array([0.25, 1.5, 3])

#建立subplot,size為(gamma變數數量*原始圖片數量)
fig, axes = plt.subplots(len(gamma) + 1,
                           len(glob.glob('HW1_test_image/*.bmp')),
                           figsize = (25, 25))

if __name__ == '__main__':
    
    #顯示原始image
    for files in glob.glob('HW1_test_image/*.bmp'):
        img = plt.imread(files)
        disp_img(0, count, img, 'original image')
        count += 1
        
    #將image做gamma transformation,共執行gamma array中元素個數
    for i in range(0, len(gamma)):
        count = 0 #紀錄現在是第count個gamma
        for files in glob.glob('HW1_test_image/*.bmp'):
    
            img = plt.imread(files)
            (dim_x, dim_y) = (img.shape[0], img.shape[1])
            
            #將原image轉換成灰階image
            gray_img = rgb2gray(img)
            
            #將轉換後的灰階image執行gamma transformation
            transformed_img = gamma_transformation(gray_img, gamma[i])
            
            #顯示經gamma transformation後的image
            disp_img(i + 1, count, transformed_img, f'transformed image ($\gamma = {gamma[i]} $)')
            count += 1 #前往下一個gamma值      
    
    plt.show()