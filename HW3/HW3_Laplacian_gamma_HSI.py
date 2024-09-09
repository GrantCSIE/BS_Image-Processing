# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:10:08 2023

@author: lab308 Shen Gwan-En
"""

def rgb2hsi(img):
     
    (dim_x, dim_y, channels) = img.shape[0], img.shape[1], img.shape[2]

    img = img / 255.0 
    
    img_B = img[:, :, 0]
    img_G = img[:, :, 1]
    img_R = img[:, :, 2]
    (dim_x, dim_y) = img.shape[0], img.shape[1]

    intensity = np.zeros((dim_x, dim_y))
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            intensity[x][y] = (img_B[x][y] + img_G[x][y] + img_R[x][y]) / 3
            
    saturation = np.zeros((dim_x, dim_y))
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            RGB_Min = min(min(img_B[x][y], img_G[x][y]), img_R[x][y])
            saturation[x][y] = 1 - (3 * (RGB_Min / (img_B[x][y] + img_G[x][y] + img_R[x][y]+ 0.000001)))
            
    hue = np.zeros((dim_x, dim_y))
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            up = 0.5 * ((img_R[x][y] - img_G[x][y]) + (img_R[x][y] - img_B[x][y]))
            low = ((img_R[x][y] - img_G[x][y]) ** 2) + (img_R[x][y] - img_B[x][y]) * (img_G[x][y] - img_B[x][y])
            if up == 0 and low == 0:
                theta = math.pi / 2
            else:
                theta = math.acos(up / math.sqrt(low))
            theta_degree = (180 / math.pi) * theta
            if img_G[x][y] >= img_B[x][y]:
                    hue[x][y] = theta_degree
            else:
                    hue[x][y] = 360 - theta_degree
                    
    hsi_img = np.zeros((dim_x, dim_y, channels))
    
    hsi_img[:, :, 0] = hue
    hsi_img[:, :, 1] = saturation
    hsi_img[:, :, 2] = intensity
    
    return hsi_img

def degree2radius(deg):
    return (math.pi / 180) * deg

def hsi2rgb(hsi_img):
        
    (dim_x, dim_y, channels) = hsi_img.shape[0], hsi_img.shape[1], hsi_img.shape[2]

    img_H = hsi_img[:, :, 0]
    img_S = hsi_img[:, :, 1]
    img_I = hsi_img[:, :, 2]
    
    img_B = np.zeros((dim_x, dim_y))
    img_G = np.zeros((dim_x, dim_y))
    img_R = np.zeros((dim_x, dim_y))
    
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            
            if img_H[x][y] >= 0 and img_H[x][y] < 120:
                img_H[x][y] = degree2radius(img_H[x][y])
                img_B[x][y] = img_I[x][y] * (1 - img_S[x][y])
                img_R[x][y] = img_I[x][y] * (1 + 
                                             (img_S[x][y] * 
                                              math.cos(img_H[x][y]) / math.cos(degree2radius(60) - img_H[x][y])))
                img_G[x][y] = 3 * img_I[x][y] - (img_R[x][y] + img_B[x][y])
                
            elif img_H[x][y] >= 120 and img_H[x][y] < 240:
                img_H[x][y] = degree2radius(img_H[x][y]) - degree2radius(120)
                img_R[x][y] = img_I[x][y] * (1 - img_S[x][y])
                img_G[x][y] = img_I[x][y] * (1 + 
                                             (img_S[x][y] * 
                                              np.cos(img_H[x][y]) / math.cos(degree2radius(60) - img_H[x][y])))
                img_B[x][y] = 3 * img_I[x][y] - (img_R[x][y] + img_G[x][y])
            
            else:
                img_H[x][y] = degree2radius(img_H[x][y]) - degree2radius(240)
                
                img_G[x][y] = img_I[x][y] * (1 - img_S[x][y])
                img_B[x][y] = img_I[x][y] * (1 + 
                                             (img_S[x][y] * 
                                              math.cos(img_H[x][y]) / math.cos(degree2radius(60) - img_H[x][y])))
                img_R[x][y] = 3 * img_I[x][y] - (img_G[x][y] + img_B[x][y])
                
    bgr_img = np.zeros((dim_x, dim_y, channels))
    
    bgr_img[:, :, 0] = img_B
    bgr_img[:, :, 1] = img_G
    bgr_img[:, :, 2] = img_R
    
    bgr_img = np.array(np.clip(bgr_img, 0, 1) * 255, dtype = 'uint8')
    
    return bgr_img

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

def Laplacian_in_spatial(gray_img, L_max, L_min):
    
    transformed_img = elementwise_multi(gray_img, Laplacian, L_min, L_max)
    sharpening_img = img_addition(gray_img, transformed_img, 1, L_max, L_min)
    
    return transformed_img, sharpening_img

def disp_img(x, y, title, img):
    axes[x][y].imshow(img)
    axes[x][y].set_title(title, fontsize = 15)
    axes[x][y].axis('off')

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

import numpy as np
import matplotlib.pyplot as plt
import glob
import math

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

        hsi_img = rgb2hsi(img)
    
        img_H = hsi_img[:, :, 0]
        img_S = hsi_img[:, :, 1]
        img_I = (hsi_img[:, :, 2]* 255).astype('uint8')
        
        img_I = gamma_transformation(img_I, threshold_bottom, threshold_top)
    
        mask_I, sharpen_img_I = Laplacian_in_spatial(img_I, L_max, L_min)
    
        sharpen_img = np.zeros((dim_x, dim_y, channels), dtype = 'float64')

    
        sharpen_img[:, :, 0] = img_H
        sharpen_img[:, :, 1] = img_S
        sharpen_img[:, :, 2] = sharpen_img_I / 255

    
        sharpen_img = hsi2rgb(sharpen_img)
        
        disp_img(0, count, 'Original image', img)
        disp_img(1, count, 'Sharping image', sharpen_img)
        count += 1
    plt.show()