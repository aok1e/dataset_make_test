# -*- coding: utf-8 -*-
#!/usr/bin/env python
import cv2
import numpy as np
import itertools

import app_colormap

#path = r"C:\aoki\labeled_img\2D_labeled_20210113_2349\Cropped\00000000_12.png" #550
#path = r"C:\aoki\labeled_img\2D_labeled_20210114_0001\Cropped\00000000_12.png" #510
path = r"C:\aoki\hand_thumb\code\code_set\pillow_save_crop.png"
center_x = 113
center_y = 113
#center_x = 320
#center_y = 240
size = 30 # size*size 偶数で
img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
print(img.dtype)

#img = img - 70
#hand_mask = cv2.inRange(img, 0, 255)
#img = cv2.bitwise_and(img, img, mask=hand_mask)
#cv2.imwrite("testdepth.png",img)

visualize_img = app_colormap.visualize_depth(img)
#cv2.imwrite("testdepthvisualize.png",visualize_img)
if not img is None:
    #h = range(-9, 10) #20*20
    h = range(int(-size/2+1), int(size/2)) #20*20
    list = [0]*size*size
    count = 0

    
    #for i, j in itertools.product(h,h):
        #list[count] = img[center_y+i, center_x+j] + 550
        #print(list[count])
        #count += 1
    

    sum = 0
    ave_num = 0
    for i in h:
        for j in h:
            list[count] = img[center_y+i, center_x+j]
            #print(str(list[count]) + " ", end='')
            print("%3d " %(list[count]), end='')
            if(list[count] != 0):
                sum = sum + list[count]
                ave_num = ave_num + 1
            count = count + 1
        print("\n", end='')
    if ave_num != 0:
        print("\nave:%f" %(sum/ave_num))

    cv2.rectangle(visualize_img, (center_x - int(size/2+1), center_y - int(size/2+1)), (center_x + int(size/2), center_y + int(size/2)), (0,0,255))
    cv2.imshow("label", visualize_img)
    cv2.waitKey(0)