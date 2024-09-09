# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:24:21 2023

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
    
    pad_height = kdim_x // 2
    pad_width = kdim_y // 2
    
    padded_image = np.zeros((dim_x + 2 * pad_height, dim_y + 2 * pad_width))
    
    padded_image[pad_height : (pad_height + dim_x), pad_width : (pad_width + dim_y)] = src_img
    
    transformed_img = np.zeros_like(src_img)
    
    for i in range(dim_x):
        for j in range(dim_y):
            sub_img = padded_image[i : (i + kdim_x), j : (j + kdim_y)]
            transformed_img[i][j] = np.sum(sub_img * kernel)
            
            if transformed_img[i][j] <= L_min:
                transformed_img[i][j] = L_min
            elif transformed_img[i][j] >= L_max:
                transformed_img[i][j] = L_max
                
    return transformed_img

def get_gaussuian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    
    kernel = kernel / np.sum(kernel)

    return kernel

def blurring(img, kdim_x, variance, L_min, L_max):
    
    kernel = get_gaussuian_kernel(kdim_x, variance)
    
    blurring_img = elementwise_multi(img, kernel, L_min, L_max)
    
    return blurring_img


def disp_img(x, y, title, img):
    axes[x][y].imshow(img, cmap = 'gray', vmin = 0, vmax = 1)
    axes[x][y].set_title(title, fontsize = 15)
    axes[x][y].axis('off')    
    
import numpy as np
import matplotlib.pyplot as plt
import glob

L_min = 0
L_max = 1
count = 0
variance = 0.2

size_test = [3, 7, 13]

path = 'HW4_test_image/'

fig, axes = plt.subplots(1 + len(size_test),
                         len(glob.glob(path + '*.jpg')),
                         figsize = (25, 25))

if __name__ == '__main__':
    for files in glob.glob(path + '*.jpg'):
        
        img = plt.imread(files)
        
        img = rgb2gray(img)
        
        img = img / 255.0
        
        disp_img(0, count, 'Original image', img)
        
        for test in range(1, len(size_test) + 1):
            
            blur_img = blurring(img, size_test[test - 1],
                                variance, L_min, L_max)        
            
            Laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype = 'int8')
    
            edge = elementwise_multi(blur_img, Laplacian, L_min, L_max)
        
            disp_img(test, count, 
                     'Edges of image with mask size = (' + str(size_test[test - 1]) + '*' + str(size_test[test - 1]) + ')',
                                                                                               edge)
        count += 1
    plt.show()
