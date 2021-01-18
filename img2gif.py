# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:43:39 2021

@author: wh
"""
from PIL import Image

#img=Image.open('img\\img (1).png')

imgs = []

for n in range(32):
    #命名规则 img (1).png
    img=Image.open('img\\img ('+str(n+1)+').png')
    imgs.append(img)

img.save('img\\new.gif',save_all=True,append_images=imgs,duration=2,)

#img.show()
