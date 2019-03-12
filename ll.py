#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.fftpack import dct

def ahash(raw):
    gray = rgb2gray(raw)
    scaled = resize(gray, (8, 8), anti_aliasing=True, mode='reflect')
    lavg = scaled.mean()
    bits = np.ravel((scaled > lavg))
    y = 0
    for i,j in enumerate(bits):
        y += j<<i
    return y

def dhash(raw):
    img = rgb2gray(raw)
    img = resize(img, (8, 9), anti_aliasing=True, mode='reflect')
    bits = []
    for i in img:
        for j in range(0,8):
            bits.append(i[j] < i[j+1])
    y = 0
    for i,j in enumerate(bits):
        y += j<<i
    return y

def phash(raw):
    img = rgb2gray(raw)
    img = resize(img, (32, 32), anti_aliasing=True, mode='reflect')
    imgdct = dct(dct(img.T, norm='ortho').T, norm='ortho')
    imgdct = imgdct[0:8,0:8]
    avg = np.ravel(imgdct)[1:].mean()
    bits = np.ravel(imgdct > avg)
    y = 0
    for i,j in enumerate(bits):
        y += j<<i
    return y

