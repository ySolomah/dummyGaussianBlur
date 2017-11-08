import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable

im = Image.open("ian.jpg");

im.save("ian2.jpg", 'JPEG', quality=12)

im = Image.open("ian.jpg");

transform = transforms.Compose([transforms.ToTensor()])
ian_img = transform(im);
print("size of ian img: " + str(ian_img.shape))
h = ian_img.shape[1]
w = ian_img.shape[2]
ian_img.resize_(1, 3, h, w)

#add_tensor = torch.FloatTensor(3, h, w).normal_(0.5, 0.15)

#ian_img = ian_img + add_tensor


#im = Image.open("ian.jpg");
#ian_img = transform(im)

print("type of ian img: " + str(type(ian_img)));
print("size of ian img: " + str(ian_img.shape));

x = torch.FloatTensor(3, 3, 9, 9).zero_();
sum = 0
for i in range (0, 9):
	for j in range (0, 9):
		for k in range (0, 3):
			x[k][k][i][j] = math.exp(4*(-math.fabs(i-4) - math.fabs(j-4)));
			#sum += math.exp(3*(-math.fabs(i-4) - math.fabs(j-4)));
			if(i == 4 and j == 4):
				x[k][k][i][j] = math.exp(0)
			sum += x[k][k][i][j]

print("sum is: " + str(sum))
sum = sum/2
for i in range (0, 9):
	for j in range (0, 9):
		for k in range (0, 3):
			x[k][k][i][j] = x[k][k][i][j]/sum


#x = torch.exp(x);
print(x);

result = torch.nn.functional.conv2d(Variable(ian_img), Variable(x), padding = 4)

print("size of output: " + str(result.shape));


torchvision.utils.save_image(transform(im) - result.data, "ian_blur.jpg")

print("hello")
