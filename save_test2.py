# -*- coding: utf-8 -*-
import cv2
import numpy as np

import copy

import app_colormap
import denoising
from PIL import Image

def make_base_depth_img(img,stand_mask):

    # 深度でカット
    img = img - 550
    #img = img - 520
    # スタンド部分をマスク処理(512×424px用のスタンド用マスクを使う)
    img = app_colormap.stand_mask(img,stand_mask)
    #距離マスク処理
    hand_mask = cv2.inRange(img, 0, 255)
    masked_imgDepth = cv2.bitwise_and(img, img, mask=hand_mask)
    return masked_imgDepth

def make_noise_removed_img(img):

    img = app_colormap.denoise(img)
    img = app_colormap.edgenoise_delete(img)

    return img

def main():
    stand_mask = cv2.imread("mask_realsense_hukusuu.png",0)     #自分
    path = "00000000.png"
    save_path = "save_test.png"
    img_org = cv2.imread(path,cv2.IMREAD_ANYDEPTH)
    img = cv2.flip(img_org, 0)

    masked_imgDepth = make_base_depth_img(img,stand_mask)
    stride = 0
    j = 0
    k = 0
    #d_outimg = masked_imgDepth[127+(stride*j):127+(stride*j)+227,207+(stride*k):207+(stride*k)+227] #もともと切り取ってた位置、切り取りが影響してるのかを調べるために最後に移動
    d_outimg = masked_imgDepth.copy()
    d_outimg = make_noise_removed_img(d_outimg)
    print("保存します2")

    im = Image.fromarray(d_outimg)
    im.crop((207+(stride*k), 127+(stride*j), 207+(stride*k)+227, 127+(stride*j)+227)).save("pillow_save_crop.png")

    cv2.imwrite("visualize_origin_size_" + save_path,app_colormap.visualize_depth(d_outimg))
    cv2.imwrite("origin_size_" + save_path,d_outimg)
    cv2.imwrite("visualize_" + save_path,app_colormap.visualize_depth(d_outimg[127+(stride*j):127+(stride*j)+227,207+(stride*k):207+(stride*k)+227]))
    cv2.imwrite(save_path, d_outimg[127+(stride*j):127+(stride*j)+227,207+(stride*k):207+(stride*k)+227])
    cv2.imshow("image",app_colormap.visualize_depth(d_outimg[127+(stride*j):127+(stride*j)+227,207+(stride*k):207+(stride*k)+227]))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()