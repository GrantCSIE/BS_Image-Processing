# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:46:02 2023

@author: Grant
"""


def rgb2gray(img):
    if img.ndim == 3: 
        (dim_x, dim_y) = img.shape[0], img.shape[1] 
        gray_img = np.zeros((dim_x, dim_y), dtype = 'uint16') 
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                gray_img[i][j] = img[i][j][0] 
        return gray_img
    
    else:
        return img

def elementwise_multi(src_img, kernel, L_min, L_max):
    
    (dim_x, dim_y) = src_img.shape[0], src_img.shape[1]
    (kdim_x, kdim_y) = kernel.shape[0], kernel.shape[1]
    
    transformed_img = np.zeros((dim_x, dim_y))
    
    for i in range(dim_x - kdim_x + 1):
        for j in range(dim_y - kdim_y + 1):
            sub_img = src_img[i : kdim_x + i, j : kdim_y + j]
            transformed_img[i][j] = np.sum(sub_img * kernel)
            
            if transformed_img[i][j] <= L_min:
                transformed_img[i][j] = L_min
            elif transformed_img[i][j] >= L_max:
                transformed_img[i][j] = L_max
                
    return transformed_img    

def img_addition(img1, img2, const, L_min, L_max):
    
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

def disp_img(x, y, title, img):
    axes[x][y].imshow(img, cmap = 'gray')
    axes[x][y].set_title(title, fontsize = 15)
    axes[x][y].axis('off')    
    

import numpy as np
import matplotlib.pyplot as plt
import glob

L_min = 0
L_max = 255
count = 0

path = 'HW4_test_image/'

fig, axes = plt.subplots(2,
                         len(glob.glob(path + '*.jpg')),
                         figsize = (25, 25))

if __name__ == '__main__':
    for files in glob.glob(path + '*.jpg'):
        
        img = plt.imread(files)
        
        img = rgb2gray(img)
        
        disp_img(0, count, 'Original image', img)
        
        Sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype = 'int8')

        Sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype = 'int8')
    
        gradient_x = elementwise_multi(img, Sobel_x, L_min, L_max)
        gradient_y = elementwise_multi(img, Sobel_x, L_min, L_max)

        gradient_x = np.abs(gradient_x)
        gradient_y = np.abs(gradient_y)
    
        edge = img_addition(gradient_x, gradient_y, 1, L_min, L_max).astype('uint8')
        
        disp_img(1, count, 'Edges of image', edge)
        
        count += 1
    plt.show()
    