# Zadanie3: Uzasadnij, że rozmiar wynikowej mapy cech wyraża się poprzez: //see "Zadanie3.png"

import torch.nn as nn
import cv2
import torchvision.transforms as transforms
# Decided that the best way to prove this would be to just compute a map twice with different dimensions

# Load the image, convert it to a tensor
img = cv2.imread("The_Sun_card_art.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2ten = transforms.Compose([transforms.ToTensor()])
img_ten = cv2ten(img)

# Create two convolutions
conv1 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=20)
conv2 = nn.Conv2d(3, 1, kernel_size=5, stride=3, padding=30)

out1 = conv1(img_ten)
out2 = conv2(img_ten)

print(f'Image size before transformation: {img.shape}')
print(f'Tensor size after transformation with parameters "3, 1, kernel_size=5, stride=1, padding=20": {out1.size()}')
print(f'Tensor size after transformation with parameters "3, 1, kernel_size=5, stride=3, padding=30": {out2.size()}')

# Output size of the transformations confirms that the formula is correct