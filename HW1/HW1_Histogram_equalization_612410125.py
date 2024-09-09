'''
function draw_histogram() 的功能為:
    繪製出image對應的histogram.
    其中,傳入值:
    index:紀錄image對應的histogram應繪製在圖表的哪一個column上,如果是原圖的histogram,畫在第二個column，如果是經轉換的histogram，畫在第三個column上.
        y:共有y = (HW1_test_image資料夾中image數) 張圖, 紀錄image對應的histogram應繪製在圖表的哪一個column上.
        src_img:欲繪製histogram的image.
        L_min, Lmax: 分別代表gray level的最小,最大值
        x_axe: 紀錄histogram的x軸範圍:(0, L - 1)
        his_title: histogram 上方的標題
    回傳值:
        p_rk:image的gray level對應的pdf
'''

def draw_histogram(index, y, src_img, L_min, L_max, x_axe, hist_title):
    
    #建立儲存0到255各個gray_value數量的容器
    p_rk = np.zeros((L_max + 1), dtype = 'float32')
    
    #統計image中有多少個pixels為某個gray level
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            p_rk[src_img[i][j]] += 1
    
    #計算gray level的pdf
    for i in range(0, p_rk.shape[0]):
        p_rk[i] = p_rk[i] / (dim_x *dim_y)
    
    #根據p_rk繪製出histogram
    axes[index][y].bar(x_axe, p_rk) 
    axes[index][y].set_title(hist_title, fontsize = fort)
    return p_rk
    
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
                        ,印出經 histogram equalization 之image時標題為: "圖片名稱.副檔名 after histogram equalization"
    回傳值:
        無.
        
'''
def disp_img(x, y, img, title):
    axes[x][y].imshow(img, cmap = 'gray',vmin = 0, vmax = 255)
    axes[x][y].set_title(title, fontsize = fort)
    axes[x][y].axis('off')

'''
function histogram_equalization() 的功能為:
    進行histogram equalization,並繪出image經histogram equalization後對應的histogram,
    並回傳image經histogram equalization後的image.
    其中,傳入值:
        y:共有y = (HW1_test_image資料夾中image數) 張圖, 紀錄經histogram equalization後的histogram應繪製在圖表的哪一個column上
        p_rk:image中每一個gray level的pdf
        img:欲進行histogram equalization的image
        L_min, Lmax: 分別代表gray level的最小,最大值
        x_axe: 紀錄histogram的x軸範圍:(0, L - 1)
        his_title: histogram 上方的標題,標題為:"histogram of 圖片名稱.副檔名 after histogram equalization"
    回傳值:
        histogram_eqed_img: 經histogram equalization的image.
'''
def histogram_equalization(y, p_rk, img, L_min, L_max, x_axe, hist_title):
    sum_of_prob_val = 0 #儲存當下累積的intensity level rk機率
    ps_sk = np.zeros((L_max + 1), dtype = 'float32')#ps(sk)
    sk = np.zeros((L_max + 1), dtype = 'float32')#sk

    if p_rk[0] == 0:
        current = 1
        #方便比較對應的值是否相同
    else:
        current = 0

    sk[0] = p_rk[0]

    for i in range(L_min + 1, L_max + 1):
        sk[i] = (p_rk[i] + sk[i - 1]) #計算sk
        if round(sk[i] * (L_max + 1), 0) == round(sk[i - 1] * (L_max + 1), 0):
            #如果rk經T轉換後對應的值sk=T(rk)和前一個r(k-1)經T轉換後對應的值一樣對到sk=T(rk)
            sum_of_prob_val += p_rk[i] #累加intensity level rk的機率值
        else: #如果rk對應到的sk和r(k-1)對應到的sk不同
            ps_sk[current] = sum_of_prob_val #儲存累積下來的intensity level rk機率值
            sum_of_prob_val = p_rk[i] #重新累積intensity level rk的機率值,從當前的index開始
            current = i
            
    
    for i in range(0, L_max + 1):
        sk[i] = round(sk[i] * (L_max), 0)
        
    
    histogram_eqed_img = np.zeros((dim_x, dim_y), dtype = 'uint8') #建立新的matrix,即將放入每個pixel經轉換的gray level
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            #原img上的gray level根據T轉換所對應的sk,assign至新的matrix相同pixel上
            histogram_eqed_img[i][j] = sk[img[i][j]] 
    return histogram_eqed_img
    

    
import numpy as np
import matplotlib.pyplot as plt
import glob
    
fort = 8 #設定圖表標題字體大小
Level = pow(2, 8) #image的gray level從0至 2^8-1
count = 0 
#紀錄histogram或圖片要放在plot的哪個位置,第1列放原始圖,第2列放原始圖對應的histogram,第3列放經轉換後的histogram,第4列放經轉換的圓片
x_axe = np.arange(start = 0, stop = Level)

#建立subplot,size為(4*原始圖片數量)
fig, axes = plt.subplots(4,
                         len(glob.glob('HW1_test_image/*.bmp')),
                         figsize = (25, 25))

    
if __name__ == '__main__':
    
    for files in glob.glob('HW1_test_image/*.bmp'):
        
        img = plt.imread(files)
        
        (dim_x, dim_y), gray_img = rgb2gray(img) 
        
        disp_img(0, count, gray_img, f"{files[len('HW1_test_image/') : ]}")
        
        p_rk = draw_histogram(1, count, gray_img, 
                              0, Level - 1, x_axe, f"histogram of {files[len('HW1_test_image/') : ]}")
        
        histogram_eqed_img = histogram_equalization(count, p_rk, gray_img, 0, Level - 1, x_axe, 
                               f"histogram of {files[len('HW1_test_image/') : ]} \n after histogram equalization")
                               
        p_rk = draw_histogram(2, count, histogram_eqed_img, 
                              0, Level - 1, x_axe, f"histogram of {files[len('HW1_test_image/') : ]} \n after histogram equalization")
        
        disp_img(3, count, histogram_eqed_img, 
                 f"{files[len('HW1_test_image/') : ]}\n after histogram equalization")
        
        count += 1
    plt.show()
        
        