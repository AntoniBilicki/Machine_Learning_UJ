# Zadanie4: Napisać funkcje do interpolacji zdjęć na dowolnie zadany wymiar. Przykładowo, mamy zdjecie 4x5,
# chcemy 10x12. Uwaga: W przypadku wielokrotności rozmiaru jest to stosunkowo proste zadanie.


# Decided to do it by hand rather than using CNN

import numpy as np
import cv2


img = cv2.imread("The_Sun_card_art.png")

def rescale(input_img, final_x, final_y):
    org_y, org_x, org_z = input_img.shape
    output_img = np.zeros((org_y, final_x, 3))
    # interpolate each row
    for i in range(org_y):
        # have to interpolate each channel separately and then merge them
        output_img[i, :, 0] = np.interp(np.linspace(0, org_x-1, final_x), range(org_x), input_img[i, :, 0])
        output_img[i, :, 1] = np.interp(np.linspace(0, org_x-1, final_x), range(org_x), input_img[i, :, 1])
        output_img[i, :, 2] = np.interp(np.linspace(0, org_x-1, final_x), range(org_x), input_img[i, :, 2])

    # interpolate each column
    input_img = output_img
    org_y, org_x, org_z = input_img.shape
    output_img = np.zeros((final_y, org_x, 3))
    for i in range(org_x):
        # have to interpolate each channel separately and then merge them
        output_img[:, i, 0] = np.interp(np.linspace(0, org_y - 1, final_y), range(org_y), input_img[:, i, 0])
        output_img[:, i, 1] = np.interp(np.linspace(0, org_y - 1, final_y), range(org_y), input_img[:, i, 1])
        output_img[:, i, 2] = np.interp(np.linspace(0, org_y - 1, final_y), range(org_y), input_img[:, i, 2])

    return np.ubyte(output_img)


rescaled_img = rescale(img, 500, 500)
print(np.array_equal(img, rescaled_img))
cv2.imshow('Base image', img)
cv2.imshow('Rescaled image', rescaled_img)
cv2.waitKey(0)
