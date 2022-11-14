# Zadanie1: Zaimplementuj filtry Sobela a następnie przetestuj na przykładowej fotografii 1D (odcienie szarości).
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as sp


img = plt.imread("The_Sun_card_art.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sobel(inputPic):

	Gx = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
	Gy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

	# decided not to reinvent the wheel and use premade convolve function, and apply our kernel to image this way
	# especially because numpy doesn't support 2d convolution
	transformedX = sp.convolve(inputPic, Gx, mode='constant', cval=1.0)
	transformedY = sp.convolve(inputPic, Gy, mode='constant', cval=1.0)

	return transformedY + transformedX



sobelxy = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
cv2.imshow('Base image', gray)
cv2.imshow('Sobel X Y using cv2.Sobel() function', sobelxy)
cv2.imshow('Sobel X Y using our own Sobel function', sobel(gray))
cv2.waitKey(0)

