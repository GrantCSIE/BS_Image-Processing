# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:17:11 2023

@author: lab308 Shen Gwan-En
"""

def img_linearlize(img):
    (dim_x, dim_y, channel) = img.shape[0], img.shape[1], img.shape[2]
    
    linearlized_img = np.zeros((dim_x, dim_y, channel), dtype = 'float32')
    
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            for c in range(0, channel):
                if img[x][y][c] <= 0.04045:
                    linearlized_img[x][y][c] = img[x][y][c] / 12.92
                else:
                    linearlized_img[x][y][c] = pow((img[x][y][c] + 0.055) / 1.055, 2.4)
                
    return linearlized_img
    
def rgb2xyz(img):
    linearlized_img = img_linearlize(img)
    
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])

    xyz_img = np.dot(linearlized_img, matrix.T)
    
    return xyz_img

def h_function(q):
    if q > 0.008856:
        return math.pow(q, 1 / 3)
    else:
        return (7.787 * q) + (16 / 116)
    
def xyz2lab(xyz_img):
    (dim_x, dim_y, channel) = xyz_img.shape[0], xyz_img.shape[1], xyz_img.shape[2]

    img_Z = xyz_img[:, :, 2]
    img_Y = xyz_img[:, :, 1]
    img_X = xyz_img[:, :, 0]
    
    X = 0.950456
    Y = 1.0
    Z = 1.088754
    
    L_star = np.zeros((dim_x, dim_y))
    a_star = np.zeros((dim_x, dim_y))
    b_star = np.zeros((dim_x, dim_y))
    
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            L_star[x][y] = (116 * h_function(img_Y[x][y] / Y)) - 16
            a_star[x][y] = 500 * (h_function(img_X[x][y] / X) - h_function(img_Y[x][y] / Y))
            b_star[x][y] = 200 * (h_function(img_Y[x][y] / Y) - h_function(img_Z[x][y] / Z))
            
            
    Lab_img = np.zeros((dim_x, dim_y, channel))
    Lab_img[:, :, 0] = L_star
    Lab_img[:, :, 1] = a_star
    Lab_img[:, :, 2] = b_star
    
    return Lab_img

def h_inv_function(q):
    if q > (6 / 29):
        return pow(q, 3)
    else:
        return ((q - 16 / 116) * 3 * pow(6 / 29, 2))
    
def lab2xyz(lab_img):
    (dim_x, dim_y, channel) = lab_img.shape[0], lab_img.shape[1], lab_img.shape[2]

    L_star = lab_img[:, :, 0]
    a_star = lab_img[:, :, 1]
    b_star = lab_img[:, :, 2]
    
    X = 0.950456
    Y = 1.0
    Z = 1.088754
    
    img_X = np.zeros((dim_x, dim_y), dtype = 'float32')
    img_Y = np.zeros((dim_x, dim_y), dtype = 'float32')
    img_Z = np.zeros((dim_x, dim_y), dtype = 'float32')
    
    for x in range(0, dim_x):
        for y in range(0, dim_y):
            
            img_Y[x][y] = (L_star[x][y] + 16.0) / 116.0
            img_X[x][y] = img_Y[x][y] + a_star[x][y] / 500.0
            img_Z[x][y] = img_Y[x][y] - b_star[x][y] / 200
            
            img_Y[x][y] = h_inv_function(img_Y[x][y]) * Y
            img_X[x][y] = h_inv_function(img_X[x][y]) * X
            img_Z[x][y] = h_inv_function(img_Z[x][y]) * Z
            
            
    xyz_img = np.zeros((dim_x, dim_y, channel))
    xyz_img[:, : ,0] = img_X
    xyz_img[:, : ,1] = img_Y
    xyz_img[:, : ,2] = img_Z
    
    return xyz_img

def img_nonlinearilze(img):
    (dim_x, dim_y, channel) = img.shape[0], img.shape[1], img.shape[2]
    
    nonlinearilzed_img = np.zeros((dim_x, dim_y, channel))
    
    for x in range(dim_x):
        for y in range(dim_y):
            for c in range(channel):
                if img[x][y][c] > 0.0031308:
                    nonlinearilzed_img[x][y][c] = 1.055 * pow(img[x][y][c], 1 / 2.4) - 0.055
                else:
                    nonlinearilzed_img[x][y][c] = 12.92 * img[x][y][c]
    
    return nonlinearilzed_img
    
def xyz2rgb(xyz_img):
    matrix = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]])

    rgb_img = np.dot(xyz_img, matrix.T)
    
    nonlinearilzed_img = img_nonlinearilze(rgb_img)
    
    nonlinearilzed_img = np.array(np.clip(nonlinearilzed_img, 0, 1) * 255, dtype = 'uint8')
    
    return nonlinearilzed_img

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
    
    print(Sum)
    
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
import math
import matplotlib.pyplot as plt
import glob

L_min = 0
L_max = 255
count = 0
threshold_bottom = 25
threshold_top = 75

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
        
        img = (img / 255.0).astype('float32')

        (dim_x, dim_y, channels) = img.shape[0], img.shape[1], img.shape[2]

        xyz_img = rgb2xyz(img)
        lab_img = xyz2lab(xyz_img)
    
        img_L = lab_img[:, :, 0]
        img_a = lab_img[:, :, 1]
        img_b = lab_img[:, :, 2]
    
        img_L = gamma_transformation(img_L, threshold_bottom, threshold_top)    
    
        L, sharped_L = Laplacian_in_spatial(img_L, L_max, L_min)
    
        sharpen_img = np.zeros((dim_x, dim_y, channels))
    
        sharpen_img[:, :, 0] = sharped_L
        sharpen_img[:, :, 1] = img_a
        sharpen_img[:, :, 2] = img_b
    
        sharpen_img = lab2xyz(sharpen_img)
        sharpen_img = xyz2rgb(sharpen_img)
    
        disp_img(0, count, 'Original image', img)
        disp_img(1, count, 'Sharping image', sharpen_img)
        
        count += 1
    
    plt.show()