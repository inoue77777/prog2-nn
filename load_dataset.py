import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

#データセットの読み込み
ds_train = datasets.FashionMNIST(root = "data",train = True, download = True)

print(f"dataset size:{len (ds_train)}")

image, target = ds_train[99]

print(type(image))
print(target)

# PIL -> torch.Tensor
image = transforms.functional.to_image(image)
print(image.shape, image.dtype)

plt.imshow(image, cmap = "gray_r", vmin = 0, vmax = 255)
plt.title(target)
plt.show()