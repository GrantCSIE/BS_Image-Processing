# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:41:57 2023

@author: lab308 Shen Gwan-En
"""
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

def img_addition(img1, img2, const, L_max, L_min):
    
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

def gamma_transformation(img, threshold_bottom, threshold_top):
    
    (dim_x, dim_y) = img.shape[0], img.shape[1]
    Sum = 0
    
    for x in range(dim_x):
        for y in range(dim_y):
            Sum += img[x][y]
    Sum = Sum / ((dim_x + dim_y) * 255)
    
    
    
    gamma_img = np.zeros((dim_x, dim_y))
    if Sum < threshold_bottom:        
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                gamma_img[i][j] = pow(1 / 255 * img[i][j], 0.5) * 255
                
    elif Sum > threshold_top:
        for i in range(0, dim_x):
            for j in range(0, dim_y):
                gamma_img[i][j] = pow(1 / 255 * img[i][j], 2) * 255
    else:
        gamma_img = img
        
    return gamma_img
        
def Laplacian_in_spatial(gray_img, L_max, L_min):
    
    transformed_img = elementwise_multi(gray_img, Laplacian, L_min, L_max)
    sharpening_img = img_addition(gray_img, transformed_img, 1, L_max, L_min)
    
    return transformed_img, sharpening_img

def disp_img(x, y, title, img):
    axes[x][y].imshow(img)
    axes[x][y].set_title(title, fontsize = 15)
    axes[x][y].axis('off')    
    
import numpy as np
import matplotlib.pyplot as plt
import glob

L_min = 0
L_max = 255
count = 0
threshold_bottom = 50
threshold_top = 200

path = 'HW3_test_image/'

fig, axes = plt.subplots(2,
                         len(glob.glob(path + '*.jpg')),
                         figsize = (25, 25))

Laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype = 'int8')


if __name__ == '__main__':
    for files in glob.glob(path + '*.jpg'):
    
        img = plt.imread(files)
            
        (dim_x, dim_y, channels) = img.shape[0], img.shape[1], img.shape[2]
            
        img_B = img[:, :, 0]
        img_G = img[:, :, 1]
        img_R = img[:, :, 2]
        
        img_B = gamma_transformation(img_B, threshold_bottom, threshold_top)
        img_G = gamma_transformation(img_G, threshold_bottom, threshold_top)
        img_R = gamma_transformation(img_R, threshold_bottom, threshold_top)        
        
        mask_B, sharpen_img_B = Laplacian_in_spatial(img_B, L_max, L_min)
        mask_G, sharpen_img_G = Laplacian_in_spatial(img_G, L_max, L_min)
        mask_R, sharpen_img_R = Laplacian_in_spatial(img_R, L_max, L_min)
            
        sharpen_img = np.zeros((dim_x, dim_y, channels), dtype = 'uint8')
            
        sharpen_img[:, :, 0] = sharpen_img_B
        sharpen_img[:, :, 1] = sharpen_img_G
        sharpen_img[:, :, 2] = sharpen_img_R
            
        disp_img(0, count, 'Original image', img)
        disp_img(1, count, 'Sharping image', sharpen_img)
            
        count += 1
    plt.show()
