#!/usr/bin/env python

import argparse
import numpy as np
from PIL import Image
import cv2
import os
import sys
import itertools
import math

import denoising as dn

def visualize_depth(im_gray16,cutFlag = False):

    if cutFlag:
        im_gray16 = im_gray16 - 550
    else :
        im_gray16 = im_gray16

    hand_mask = cv2.inRange(im_gray16, 0, 255) #二値化処理
    img = cv2.bitwise_and(im_gray16, im_gray16, mask=hand_mask)
    img = cv2.normalize(im_gray16,im_gray16,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
    # img = cv2.normalize(img,img,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
    
    img = np.array(img * 255,dtype = np.uint8)
    #ガンマ補正
    # gamma = 20
    # lookUpTable = np.zeros((256, 1), dtype = 'uint8')
    # for i in range(256):
    #     #lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma) #ガンマ
    #     lookUpTable[i][0] = 255 / (1+math.exp(-gamma*(i-170)/255)) #シグモイド
    # im_color = cv2.LUT(img, lookUpTable)

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE) #モノクロ画像に疑似的に色を付ける
    #im_color = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW) #モノクロ画像に疑似的に色を付ける

    return im_color

def morph_mask(im_gray16,cutFlag = False,colorFlag = False):

    if cutFlag:
        im_gray16 = im_gray16 - 550
    else :
        im_gray16 = im_gray16

    img = dn.denoise(im_gray16)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)    

    if(colorFlag == True):
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)

    return img

def stand_mask(im_gray16,stand_mask):
    
    # w,h = stand_mask.shape
    # im_gray16 = im_gray16[int(h/2 - 227/2):int(h/2 + 227/2),int(w/2 - 227/2):int(w/2 + 227/2)]
    # img_d = dn.denoise(im_gray16)
    img = cv2.bitwise_and(im_gray16, im_gray16, mask=stand_mask)
    # print(img.dtype) #16bit
    return img

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
    
    # cv2.imshow("erode",erode.astype(np.uint8))
    erosion_mask = cv2.erode(erode.astype(np.uint8),kernel,iterations = 1)
    
    masked_imgDepth = cv2.bitwise_and(img, img, mask=erosion_mask)

    return masked_imgDepth

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Convert 16-bit Depth image to color')
    parser.add_argument('dimg',help='16-bit depth image')
    # parser.add_argument('--out', '-o', default='colored',help='output image name')
    args = parser.parse_args()
    
    

    im_gray = cv2.imread(args.dimg,cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread("X:\hand_thumb\code\stand_mask_full.png",0)
    print(im_gray.shape)
    im_gray = im_gray - 550
    im_gray = stand_mask(im_gray,mask)
    
    hand_mask = cv2.inRange(im_gray, 0, 255)
    img = cv2.bitwise_and(im_gray, im_gray, mask=hand_mask)

    img = denoise(img)
    img = edgenoise_delete(img)
    
    im_color = visualize_depth(img)

    cv2.imshow("colored.png",im_color)
    cv2.waitKey(0)