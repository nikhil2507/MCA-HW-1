import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import json

def LoG(sigma):
    
    win = np.ceil(sigma*6)
    y,x = np.ogrid[-win//2:win//2+1, -win//2:win//2+1]

    x_fil = np.exp(-(x**2/(2.*(sigma**2))))
    y_fil = np.exp(-(y**2/(2.*(sigma**2))))
    sum_fil = (-(2*sigma**2) + (x**2 + y**2) ) * (x_fil*y_fil) * (1/(2*np.pi*sigma**4)) 
    return sum_fil


def LoG_conv(image, th):
    imgs = []
    coords = []
    (height, width) = img.shape
    for i in range(0, 5):
        sig = sigma*(np.power(k,i))
        fil = LoG(sig)
        img2 = cv2.filter2D(img, -1, fil)
        img2 = np.pad(img2, ((1,1),(1,1)), 'constant')
        img2 = np.square(img2)
        imgs.append(img2)
    imgs_np = np.array([i for i in imgs])
    # print(imgs_np)
    for i in range(1,height):
        for j in range(1, width):
            slice_img = imgs_np[:,i:i+3,j:j+3]
            max = np.amax(slice_img) 
            if ( max >= th ):
                z,x,y = np.unravel_index(slice_img.argmax(), slice_img.shape)
                coords.append((i+x-1,j+y-1,k**z*sigma))
    return coords 

points = {}

for f in os.listdir('images'):
    f = (str)(f)
    img = cv2.imread(f,0)
    img = cv2.resize(img, (64,64))
    k = 1.414
    sigma = 1.2
    img = img/255.0 

    # imgs_np = LoG_conv(img)
    coords = list(set(LoG_conv(img, th = 0.03)))
    # fig, axis = plt.subplots()
    # axis.imshow(img)
    name = f
    print(name)
    for blob in coords:
        y,x,r = blob
        # c = plt.Circle((x,y), r*k, color='red', linewidth=0.7, fill=False)
        # axis.add_patch(c)
        x = (int)(x)
        y = (int)(y)
        if name in points:
            points[name].append((x,y))
        else:
            points[name] = [(x,y)]
    # plt.show()    

with open('data.json', 'w') as outfile:
    json.dump(points, outfile)