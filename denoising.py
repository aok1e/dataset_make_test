# -*- coding: utf-8 -*-

import sys
import os
import io

import cv2
import numpy as np
from PIL import Image
import glob
import app_colormap
import name_pattern
import itertools

def mean_has_number(img,center):
    
    mean = 0
    num = 0
    
    for x,y in itertools.product(range(-2,3),range(-2,3)):
        if (center[0] + x < 227 and center[1] + y < 227):
            add = img[center[1] + y,center[0] + x]
            if add > 0:
                mean = mean + add
                num = num + 1
          
    if(num > 0):
        return int(round(mean/num)) 
    else:
        return 0

def denoise(img):
    
    height,width = img.shape[:2]

    ret,labels = cv2.threshold(img.astype(np.uint8),1,255,cv2.THRESH_BINARY_INV)
    nlabels,labelimg,contours,CoGs = cv2.connectedComponentsWithStats(labels)

    for i in range(2,nlabels):
        if(contours[i][4] <= 1000):
            #画素有るところの平均値
            mean = mean_has_number(img,[int(CoGs[i][0]),int(CoGs[i][1])])
            # print(mean)
            for y,x in itertools.product(range(0,height),range(0,width)):
                if labelimg[y,x] == i:
                    # img_color[y,x] = [255,0,255]
                    img[y,x] = mean

    return img                    

def edgenoise_delete(img):
    
    # マスク作成
    mimg = img.copy()
    # mimg -= 550 #stand_maskedはすでに引いてあるから要らない
    kernel = np.ones((2,2),np.uint8)
    ret,erode = cv2.threshold(mimg,0,255,cv2.THRESH_BINARY)
    
    cv2.imshow("erode",erode.astype(np.uint8))
    erosion_mask = cv2.erode(erode.astype(np.uint8),kernel,iterations = 1)
    
    masked_imgDepth = cv2.bitwise_and(img, img, mask=erosion_mask)

    return masked_imgDepth

def main():

    path = sys.argv[1]
    img = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
    cv2.imshow("img",app_colormap.visualize_depth(img))
    res = edgenoise_delete(img)
    cv2.imshow("res",app_colormap.visualize_depth(res))
    cv2.waitKey(0)
    # cv2.imwrite("edgedeletetest_src.png",app_colormap.visualize_depth(img))
    # cv2.imwrite("edgedeletetest_res.png",app_colormap.visualize_depth(res))

    # label_txt = sys.argv[1]
    # file_num = 0

    # savedir = os.path.join("labeled_img","denoise_and_standcut" + name_pattern.add_datetime())
    # os.makedirs(savedir,exist_ok=True)

    # with open(label_txt, 'r') as f:

    #     for line in f:

    #         sys.stderr.write('{}\r'.format(file_num + 1))
    #         sys.stderr.flush()
            
    #         lines = line.split()
    #         path = lines[0]  # depthのフルパス

    #         img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    #         img_color = app_colormap.visualize_depth(img)


    #         # for i,j in itertools.product([x*5 for x in range(width//5)],[x*5 for x in range(height//5)]):#バッチ走査
                
    #         #     min_img = img[j:j + 5,i:i + 5] #バッチ

    #         #     spot_index = np.where(min_img == 0)#バッチ内インデックス
    #         #     num = len(np.where(min_img == 0)[0])#ノイズの数
    #         #     if(num >= 0 and num <= 8):
    #         #         mean = np.sum(min_img) // (25 - num) 
    #         #         for s in range(num):
    #         #             sX = spot_index[1][s] + i
    #         #             sY = spot_index[0][s] + j
    #         #             # print(sX,sY)
    #         #             img_color[sY,sX] = [255,0,255]
    #         masked_img = app_colormap.stand_mask(img,"X:\hand_thumb\code\stand_mask.png")
    #         img = denoise(masked_img)

    #         # cv2.imshow("label",labels)
    #         # cv2.imshow("res_label",img_color)
    #         cv2.imshow("res_img",app_colormap.visualize_depth(img))
    #         cv2.imwrite(os.path.join(savedir,os.path.basename(path)),img)
    #         # cv2.waitKey(0)

    #         file_num = file_num + 1

    # f.close()


if __name__ == '__main__':
    main()
